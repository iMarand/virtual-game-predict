import json
import os
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, confusion_matrix
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import pickle
import base64
from datetime import datetime, timedelta
import warnings
from collections import defaultdict
warnings.filterwarnings('ignore')

class EnhancedSportsPredictor:
    def __init__(self, train_folder='train', results_folder='previousMatch'):
        self.train_folder = train_folder
        self.results_folder = results_folder
        self.models = {}
        self.encoders = {}
        self.scalers = {}
        self.feature_columns = []
        self.training_data = pd.DataFrame()
        self.results_data = pd.DataFrame()
        self.matched_data = pd.DataFrame()  # Data with both odds and results
        self.upcoming_data = pd.DataFrame()  # New: upcoming matches data
        self.team_stats = defaultdict(dict)
        self.h2h_stats = defaultdict(dict)
        self.fill_values = {}
        self.prediction_accuracy = defaultdict(list)  # Track prediction accuracy over time
        
    def decode_probability(self, prob_string):
        """Decode the base64 encoded probability string"""
        try:
            decoded = base64.b64decode(prob_string)
            prob_data = json.loads(decoded.decode('utf-8'))
            
            win_value = prob_data.get('win', 0)
            refund_value = prob_data.get('refund', 0)
            key_value = prob_data.get('key', 0)
            
            total = abs(win_value) + abs(refund_value) + abs(key_value) + 1e-10
            prob = abs(win_value) / total
            prob = max(0.001, min(0.999, prob))
            
            return {
                'probability': prob,
                'win_value': win_value,
                'refund_value': refund_value,
                'key_value': key_value
            }
        except Exception as e:
            return {
                'probability': 0.5,
                'win_value': 0,
                'refund_value': 0,
                'key_value': 0
            }
    
    def load_upcoming_data(self, filename):
        """Load upcoming match data from specified JSON file"""
        filepath = os.path.join(self.train_folder, filename)
        
        if not os.path.exists(filepath):
            print(f"Upcoming data file {filepath} not found.")
            return pd.DataFrame()
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            matches = data.get('items', [])
            upcoming_matches = []
            
            for match in matches:
                features = self.extract_features_from_match(match)
                # Mark as upcoming match
                features['is_upcoming'] = True
                upcoming_matches.append(features)
            
            df = pd.DataFrame(upcoming_matches)
            print(f"Loaded {len(df)} upcoming matches from {filename}")
            return df
            
        except Exception as e:
            print(f"Error loading upcoming data from {filename}: {e}")
            return pd.DataFrame()
    
    def find_match_in_upcoming(self, home_team, away_team, league, upcoming_df):
        """Find specific match in upcoming data"""
        if upcoming_df.empty:
            return None
        
        # Try exact match first
        exact_match = upcoming_df[
            (upcoming_df['home_team'].str.upper() == home_team.upper()) &
            (upcoming_df['away_team'].str.upper() == away_team.upper()) &
            (upcoming_df['league'].str.upper() == league.upper())
        ]
        
        if not exact_match.empty:
            return exact_match.iloc[0]
        
        # Try partial match (team names might be slightly different)
        partial_match = upcoming_df[
            (upcoming_df['home_team'].str.contains(home_team[:3], case=False, na=False)) &
            (upcoming_df['away_team'].str.contains(away_team[:3], case=False, na=False)) &
            (upcoming_df['league'].str.contains(league[:5], case=False, na=False))
        ]
        
        if not partial_match.empty:
            print(f"Found partial match: {partial_match.iloc[0]['home_team']} vs {partial_match.iloc[0]['away_team']}")
            return partial_match.iloc[0]
        
        return None
    
    def predict_with_upcoming_data(self, home_team, away_team, league, upcoming_filename=None):
        """Enhanced prediction using upcoming match data with current odds"""
        if not self.models:
            print("No trained models available. Please train first.")
            return None
        
        upcoming_match_data = None
        
        # Load upcoming data if filename provided
        if upcoming_filename:
            upcoming_df = self.load_upcoming_data(upcoming_filename)
            if not upcoming_df.empty:
                upcoming_match_data = self.find_match_in_upcoming(home_team, away_team, league, upcoming_df)
        
        if upcoming_match_data is not None:
            print(f"Using current odds data from {upcoming_filename}")
            
            # Use the actual upcoming match data with current odds
            match_data = upcoming_match_data.to_dict()
            
            # Add derived features
            derived = self.calculate_derived_features(match_data)
            match_data.update(derived)
            
            # Convert to DataFrame and add team features
            match_df = pd.DataFrame([match_data])
            match_df = self.add_enhanced_team_features(match_df)
            
            # Prepare features for prediction
            X, _ = self.prepare_features(match_df, is_training=False)
            
            # Enhanced prediction with current odds integration
            predictions = {}
            probabilities = {}
            confidence_scores = {}
            odds_analysis = {}
            
            for target_name, model in self.models.items():
                try:
                    pred = model.predict(X)[0]
                    
                    if hasattr(model, 'predict_proba'):
                        pred_proba = model.predict_proba(X)[0]
                        model_confidence = max(pred_proba)
                        
                        # Compare model prediction with current odds
                        if target_name == 'winner':
                            # Get current odds probabilities
                            current_odds = [
                                match_data.get('away_win_odds', 2.0),
                                match_data.get('draw_odds', 3.0),
                                match_data.get('home_win_odds', 2.0)
                            ]
                            current_probs = [1/odd for odd in current_odds]
                            total_prob = sum(current_probs)
                            normalized_probs = [p/total_prob for p in current_probs]
                            
                            # Calculate agreement between model and market
                            model_prob_dist = pred_proba
                            market_prob_dist = np.array(normalized_probs)
                            
                            # KL divergence to measure agreement
                            kl_div = np.sum(model_prob_dist * np.log(model_prob_dist / (market_prob_dist + 1e-10) + 1e-10))
                            agreement_score = 1 / (1 + kl_div)  # Convert to 0-1 scale
                            
                            odds_analysis[target_name] = {
                                'model_probs': model_prob_dist.tolist(),
                                'market_probs': normalized_probs,
                                'agreement_score': agreement_score,
                                'value_bet': pred != np.argmax(normalized_probs),  # Model disagrees with market
                                'current_odds': current_odds
                            }
                            
                            # Adjust confidence based on agreement and historical performance
                            if target_name in self.prediction_accuracy:
                                historical_accuracy = np.mean(self.prediction_accuracy[target_name])
                                adjusted_confidence = model_confidence * historical_accuracy * agreement_score
                            else:
                                adjusted_confidence = model_confidence * agreement_score
                        
                        elif target_name.startswith('over_under'):
                            # Handle over/under predictions
                            goal_line = target_name.split('_')[-1]
                            over_odds = match_data.get(f'over_{goal_line}_odds', 2.0)
                            under_odds = match_data.get(f'under_{goal_line}_odds', 2.0)
                            
                            market_over_prob = (1/over_odds) / ((1/over_odds) + (1/under_odds))
                            model_over_prob = pred_proba[1] if len(pred_proba) > 1 else 0.5
                            
                            agreement = 1 - abs(market_over_prob - model_over_prob)
                            adjusted_confidence = model_confidence * agreement
                            
                            odds_analysis[target_name] = {
                                'model_over_prob': model_over_prob,
                                'market_over_prob': market_over_prob,
                                'agreement_score': agreement,
                                'value_bet': abs(model_over_prob - market_over_prob) > 0.1,
                                'over_odds': over_odds,
                                'under_odds': under_odds
                            }
                        
                        elif target_name == 'btts':
                            # Handle BTTS predictions
                            btts_yes_odds = match_data.get('btts_yes_odds', 2.0)
                            btts_no_odds = match_data.get('btts_no_odds', 2.0)
                            
                            market_yes_prob = (1/btts_yes_odds) / ((1/btts_yes_odds) + (1/btts_no_odds))
                            model_yes_prob = pred_proba[1] if len(pred_proba) > 1 else 0.5
                            
                            agreement = 1 - abs(market_yes_prob - model_yes_prob)
                            adjusted_confidence = model_confidence * agreement
                            
                            odds_analysis[target_name] = {
                                'model_yes_prob': model_yes_prob,
                                'market_yes_prob': market_yes_prob,
                                'agreement_score': agreement,
                                'value_bet': abs(model_yes_prob - market_yes_prob) > 0.1,
                                'yes_odds': btts_yes_odds,
                                'no_odds': btts_no_odds
                            }
                        
                        else:
                            adjusted_confidence = model_confidence
                    else:
                        pred_proba = None
                        adjusted_confidence = 0.5
                    
                    predictions[target_name] = pred
                    probabilities[target_name] = pred_proba
                    confidence_scores[target_name] = adjusted_confidence
                    
                except Exception as e:
                    print(f"Error predicting {target_name}: {e}")
            
            return {
                'predictions': predictions,
                'probabilities': probabilities,
                'confidence': confidence_scores,
                'odds_analysis': odds_analysis,
                'match_info': {
                    'home_team': home_team,
                    'away_team': away_team,
                    'league': league,
                    'using_current_odds': True
                },
                'enhanced_analysis': self.get_enhanced_match_analysis(home_team, away_team, league),
                'current_odds_data': {
                    'home_win_odds': match_data.get('home_win_odds'),
                    'away_win_odds': match_data.get('away_win_odds'),
                    'draw_odds': match_data.get('draw_odds'),
                    'source_file': upcoming_filename
                }
            }
        else:
            # Fallback to regular prediction without current odds
            print("No upcoming odds data found. Using historical data prediction...")
            return self.predict_match(home_team, away_team, league)
    
    def predict_all_upcoming_matches(self, upcoming_filename):
        """Predict all matches in the upcoming data file"""
        upcoming_df = self.load_upcoming_data(upcoming_filename)
        
        if upcoming_df.empty:
            print(f"No upcoming matches found in {upcoming_filename}")
            return []
        
        print(f"\n=== PREDICTING ALL MATCHES FROM {upcoming_filename} ===")
        
        all_predictions = []
        
        for idx, match in upcoming_df.iterrows():
            home_team = match['home_team']
            away_team = match['away_team']
            league = match['league']
            
            print(f"\nPredicting: {home_team} vs {away_team} ({league})")
            
            # Use the match data directly since it already has current odds
            match_data = match.to_dict()
            
            # Add derived features
            derived = self.calculate_derived_features(match_data)
            match_data.update(derived)
            
            # Convert to DataFrame and add team features
            match_df = pd.DataFrame([match_data])
            match_df = self.add_enhanced_team_features(match_df)
            
            # Prepare features for prediction
            X, _ = self.prepare_features(match_df, is_training=False)
            
            if X.empty:
                print(f"  Could not prepare features for this match")
                continue
            
            # Make predictions
            match_predictions = {}
            match_confidence = {}
            odds_analysis = {}
            
            for target_name, model in self.models.items():
                try:
                    pred = model.predict(X)[0]
                    
                    if hasattr(model, 'predict_proba'):
                        pred_proba = model.predict_proba(X)[0]
                        confidence = max(pred_proba)
                        
                        # Analyze against current odds
                        if target_name == 'winner':
                            current_odds = [
                                match_data.get('away_win_odds', 2.0),
                                match_data.get('draw_odds', 3.0),
                                match_data.get('home_win_odds', 2.0)
                            ]
                            current_probs = [1/odd for odd in current_odds]
                            total_prob = sum(current_probs)
                            normalized_probs = [p/total_prob for p in current_probs]
                            
                            # Check for value bets
                            model_winner = pred
                            market_winner = np.argmax(normalized_probs)
                            is_value_bet = model_winner != market_winner
                            
                            odds_analysis[target_name] = {
                                'model_winner': ['Away', 'Draw', 'Home'][model_winner],
                                'market_favorite': ['Away', 'Draw', 'Home'][market_winner],
                                'is_value_bet': is_value_bet,
                                'model_confidence': confidence,
                                'odds': dict(zip(['away', 'draw', 'home'], current_odds))
                            }
                    else:
                        confidence = 0.5
                    
                    match_predictions[target_name] = pred
                    match_confidence[target_name] = confidence
                    
                except Exception as e:
                    print(f"    Error predicting {target_name}: {e}")
            
            # Format prediction results
            prediction_result = {
                'match': f"{home_team} vs {away_team}",
                'league': league,
                'start_time': match_data.get('start_time', ''),
                'predictions': match_predictions,
                'confidence': match_confidence,
                'odds_analysis': odds_analysis,
                'current_odds': {
                    'home_win': match_data.get('home_win_odds'),
                    'away_win': match_data.get('away_win_odds'),
                    'draw': match_data.get('draw_odds')
                }
            }
            
            all_predictions.append(prediction_result)
            
            # Display quick summary
            if 'winner' in match_predictions:
                winner_map = {0: away_team, 1: 'Draw', 2: home_team}
                predicted_winner = winner_map.get(match_predictions['winner'], 'Unknown')
                winner_confidence = match_confidence.get('winner', 0)
                
                print(f"  Predicted Winner: {predicted_winner} (Confidence: {winner_confidence:.1%})")
                
                if 'winner' in odds_analysis and odds_analysis['winner']['is_value_bet']:
                    print(f"  🎯 VALUE BET DETECTED - Model disagrees with market!")
                    print(f"     Model: {odds_analysis['winner']['model_winner']}")
                    print(f"     Market: {odds_analysis['winner']['market_favorite']}")
        
        return all_predictions
    
    def display_upcoming_predictions_summary(self, predictions):
        """Display a comprehensive summary of all upcoming predictions"""
        if not predictions:
            print("No predictions to display")
            return
        
        print(f"\n{'='*60}")
        print(f"UPCOMING MATCHES PREDICTION SUMMARY ({len(predictions)} matches)")
        print(f"{'='*60}")
        
        # Group by confidence levels
        high_confidence = []
        medium_confidence = []
        low_confidence = []
        value_bets = []
        
        for pred in predictions:
            winner_confidence = pred['confidence'].get('winner', 0)
            
            if winner_confidence > 0.75:
                high_confidence.append(pred)
            elif winner_confidence > 0.60:
                medium_confidence.append(pred)
            else:
                low_confidence.append(pred)
            
            # Check for value bets
            if ('winner' in pred['odds_analysis'] and 
                pred['odds_analysis']['winner'].get('is_value_bet', False)):
                value_bets.append(pred)
        
        # Display by confidence categories
        categories = [
            ("HIGH CONFIDENCE PREDICTIONS (>75%)", high_confidence),
            ("MEDIUM CONFIDENCE PREDICTIONS (60-75%)", medium_confidence),
            ("LOW CONFIDENCE PREDICTIONS (<60%)", low_confidence)
        ]
        
        for category_name, category_predictions in categories:
            if category_predictions:
                print(f"\n{category_name}:")
                print("-" * 50)
                
                for pred in category_predictions:
                    winner_pred = pred['predictions'].get('winner', -1)
                    winner_map = {0: pred['match'].split(' vs ')[1], 1: 'Draw', 2: pred['match'].split(' vs ')[0]}
                    predicted_winner = winner_map.get(winner_pred, 'Unknown')
                    confidence = pred['confidence'].get('winner', 0)
                    
                    print(f"  {pred['match']} ({pred['league']})")
                    print(f"    Prediction: {predicted_winner} ({confidence:.1%})")
                    
                    # Show current odds
                    if pred['current_odds']['home_win']:
                        home_team = pred['match'].split(' vs ')[0]
                        away_team = pred['match'].split(' vs ')[1]
                        print(f"    Current Odds: {home_team} {pred['current_odds']['home_win']:.2f} | "
                              f"Draw {pred['current_odds']['draw']:.2f} | "
                              f"{away_team} {pred['current_odds']['away_win']:.2f}")
                    print()
        
        # Display value bets separately
        if value_bets:
            print(f"\n🎯 VALUE BET OPPORTUNITIES ({len(value_bets)} found):")
            print("-" * 50)
            
            for pred in value_bets:
                winner_pred = pred['predictions'].get('winner', -1)
                winner_map = {0: pred['match'].split(' vs ')[1], 1: 'Draw', 2: pred['match'].split(' vs ')[0]}
                predicted_winner = winner_map.get(winner_pred, 'Unknown')
                confidence = pred['confidence'].get('winner', 0)
                
                odds_info = pred['odds_analysis']['winner']
                
                print(f"  {pred['match']} ({pred['league']})")
                print(f"    Model Prediction: {odds_info['model_winner']} ({confidence:.1%})")
                print(f"    Market Favorite: {odds_info['market_favorite']}")
                print(f"    Potential Value: Model sees {odds_info['model_winner']} as more likely")
                print()
        
        # Overall statistics
        total_matches = len(predictions)
        avg_confidence = np.mean([pred['confidence'].get('winner', 0) for pred in predictions])
        
        print(f"\nOVERALL STATISTICS:")
        print(f"  Total Matches: {total_matches}")
        print(f"  Average Confidence: {avg_confidence:.1%}")
        print(f"  High Confidence: {len(high_confidence)} ({len(high_confidence)/total_matches:.1%})")
        print(f"  Value Bet Opportunities: {len(value_bets)} ({len(value_bets)/total_matches:.1%})")
    
    def extract_match_result(self, result_data):
        """Extract actual match results from previousMatch JSON data"""
        features = {}
        
        # Basic match info
        features['match_id'] = result_data['id']
        features['home_team'] = result_data['participants'][0]['name']
        features['away_team'] = result_data['participants'][1]['name']
        features['league'] = result_data['competition']['name']
        features['region'] = result_data['region']['name']
        features['start_time'] = result_data['startTime']
        
        # Extract actual scores
        home_participant = None
        away_participant = None
        
        for participant_result in result_data['results']['participantPeriodResults']:
            participant_type = participant_result['participant']['type']
            
            for period_result in participant_result['periodResults']:
                period_name = period_result['period']['slug']
                
                if period_name == 'FULL_TIME_EXCLUDING_OVERTIME':
                    score = int(period_result['result'])
                    
                    if participant_type == 'HOME':
                        features['actual_home_score'] = score
                        home_participant = participant_result['participant']['id']
                    elif participant_type == 'AWAY':
                        features['actual_away_score'] = score
                        away_participant = participant_result['participant']['id']
        
        # Determine actual winner
        if 'actual_home_score' in features and 'actual_away_score' in features:
            home_score = features['actual_home_score']
            away_score = features['actual_away_score']
            
            if home_score > away_score:
                features['actual_winner'] = 2  # Home win
            elif away_score > home_score:
                features['actual_winner'] = 0  # Away win
            else:
                features['actual_winner'] = 1  # Draw
            
            # Calculate total goals
            features['actual_total_goals'] = home_score + away_score
            
            # Both teams to score
            features['actual_btts'] = 1 if (home_score > 0 and away_score > 0) else 0
            
            # Over/Under results for different lines
            for line in [1.5, 2.5, 3.5]:
                features[f'actual_over_{line}'] = 1 if features['actual_total_goals'] > line else 0
        
        return features
    
    def load_results_data(self):
        """Load all match results from previousMatch folder"""
        all_results = []
        
        if not os.path.exists(self.results_folder):
            print(f"Results folder {self.results_folder} not found.")
            return pd.DataFrame()
        
        json_files = [f for f in os.listdir(self.results_folder) if f.endswith('.json')]
        
        if not json_files:
            print(f"No JSON files found in {self.results_folder}")
            return pd.DataFrame()
        
        print(f"Loading results from {len(json_files)} files...")
        
        for filename in json_files:
            filepath = os.path.join(self.results_folder, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    
                matches_in_file = data.get('items', [])
                
                for match in matches_in_file:
                    result_features = self.extract_match_result(match)
                    all_results.append(result_features)
                    
            except Exception as e:
                print(f"Error loading results from {filename}: {e}")
        
        df = pd.DataFrame(all_results)
        print(f"Loaded {len(df)} match results")
        return df
    
    def normalize_start_time(self, start_time_str):
        """Normalize start time for matching (handle slight time differences)"""
        try:
            dt = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
            # Round to nearest 5 minutes to handle slight timing differences
            minutes = (dt.minute // 5) * 5
            normalized = dt.replace(minute=minutes, second=0, microsecond=0)
            return normalized.isoformat()
        except:
            return start_time_str
    
    def match_odds_with_results(self):
        """Match training data (odds) with actual results using startTime and teams"""
        print("Matching odds data with actual results...")
        
        if self.training_data.empty or self.results_data.empty:
            print("Missing training data or results data for matching")
            return pd.DataFrame()
        
        matched_records = []
        match_count = 0
        
        # Create lookup dictionary for faster matching
        results_lookup = {}
        for _, result in self.results_data.iterrows():
            normalized_time = self.normalize_start_time(result['start_time'])
            key = f"{result['home_team']}_{result['away_team']}_{normalized_time}"
            results_lookup[key] = result
        
        print(f"Created lookup for {len(results_lookup)} result records")
        
        for _, odds_match in self.training_data.iterrows():
            normalized_time = self.normalize_start_time(odds_match['start_time'])
            
            # Try exact match first
            key = f"{odds_match['home_team']}_{odds_match['away_team']}_{normalized_time}"
            
            if key in results_lookup:
                result_match = results_lookup[key]
                match_count += 1
            else:
                # Try with time tolerance (±10 minutes)
                odds_dt = datetime.fromisoformat(normalized_time.replace('Z', '+00:00'))
                result_match = None
                
                for result_key, result_data in results_lookup.items():
                    result_teams = result_key.split('_')
                    if (len(result_teams) >= 3 and 
                        result_teams[0] == odds_match['home_team'] and 
                        result_teams[1] == odds_match['away_team']):
                        
                        result_time_str = '_'.join(result_teams[2:])
                        try:
                            result_dt = datetime.fromisoformat(result_time_str.replace('Z', '+00:00'))
                            time_diff = abs((odds_dt - result_dt).total_seconds())
                            
                            if time_diff <= 600:  # 10 minutes tolerance
                                result_match = result_data
                                match_count += 1
                                break
                        except:
                            continue
            
            if result_match is not None:
                # Combine odds data with actual results
                combined_record = odds_match.to_dict()
                
                # Add actual results
                combined_record.update({
                    'actual_home_score': result_match['actual_home_score'],
                    'actual_away_score': result_match['actual_away_score'],
                    'actual_winner': result_match['actual_winner'],
                    'actual_total_goals': result_match['actual_total_goals'],
                    'actual_btts': result_match['actual_btts']
                })
                
                # Add over/under actual results
                for line in [1.5, 2.5, 3.5]:
                    combined_record[f'actual_over_{line}'] = result_match[f'actual_over_{line}']
                
                matched_records.append(combined_record)
        
        matched_df = pd.DataFrame(matched_records)
        print(f"Successfully matched {match_count} odds records with results")
        print(f"Match rate: {match_count/len(self.training_data)*100:.1f}%")
        
        return matched_df
    
    def calculate_prediction_accuracy_features(self, df):
        """Calculate features based on prediction accuracy patterns"""
        print("Calculating prediction accuracy features...")
        
        accuracy_features = []
        
        for _, match in df.iterrows():
            features = {}
            
            # Odds vs actual outcome analysis
            if all(col in match for col in ['home_win_odds', 'away_win_odds', 'draw_odds', 'actual_winner']):
                # Calculate implied probabilities
                home_implied = 1 / match['home_win_odds']
                away_implied = 1 / match['away_win_odds']
                draw_implied = 1 / match['draw_odds']
                
                total_implied = home_implied + away_implied + draw_implied
                if total_implied > 0:
                    home_norm_prob = home_implied / total_implied
                    away_norm_prob = away_implied / total_implied
                    draw_norm_prob = draw_implied / total_implied
                    
                    # Bookmaker accuracy
                    actual_winner = int(match['actual_winner'])
                    if actual_winner == 2:  # Home win
                        features['bookmaker_accuracy'] = home_norm_prob
                    elif actual_winner == 0:  # Away win
                        features['bookmaker_accuracy'] = away_norm_prob
                    else:  # Draw
                        features['bookmaker_accuracy'] = draw_norm_prob
                    
                    # Surprise factor (how unexpected was the result)
                    probs = [away_norm_prob, draw_norm_prob, home_norm_prob]
                    features['surprise_factor'] = 1 - probs[actual_winner]
                    
                    # Favorite won indicator
                    favorite_outcome = np.argmax(probs)
                    features['favorite_won'] = 1 if favorite_outcome == actual_winner else 0
            
            # Over/Under accuracy
            for line in [1.5, 2.5, 3.5]:
                over_prob_col = f'over_{line}_prob'
                actual_over_col = f'actual_over_{line}'
                
                if over_prob_col in match and actual_over_col in match:
                    over_prob = match[over_prob_col]
                    actual_over = match[actual_over_col]
                    
                    # Bookmaker O/U accuracy
                    if over_prob > 0.5 and actual_over == 1:
                        features[f'ou_{line}_accuracy'] = over_prob
                    elif over_prob <= 0.5 and actual_over == 0:
                        features[f'ou_{line}_accuracy'] = 1 - over_prob
                    else:
                        features[f'ou_{line}_accuracy'] = 0.5 - abs(over_prob - 0.5)
            
            # BTTS accuracy
            if 'btts_yes_prob' in match and 'actual_btts' in match:
                btts_prob = match['btts_yes_prob']
                actual_btts = match['actual_btts']
                
                if btts_prob > 0.5 and actual_btts == 1:
                    features['btts_accuracy'] = btts_prob
                elif btts_prob <= 0.5 and actual_btts == 0:
                    features['btts_accuracy'] = 1 - btts_prob
                else:
                    features['btts_accuracy'] = 0.5 - abs(btts_prob - 0.5)
            
            accuracy_features.append(features)
        
        return pd.DataFrame(accuracy_features)
    
    def build_enhanced_team_statistics(self, df):
        """Build team statistics including actual performance data"""
        print("Building enhanced team statistics with actual results...")
        
        # Reset statistics
        self.team_stats = defaultdict(lambda: {
            'total_matches': 0, 'wins': 0, 'draws': 0, 'losses': 0,
            'home_matches': 0, 'home_wins': 0, 'home_draws': 0, 'home_losses': 0,
            'away_matches': 0, 'away_wins': 0, 'away_draws': 0, 'away_losses': 0,
            'goals_scored': [], 'goals_conceded': [], 'total_goals': [],
            'avg_home_odds': [], 'avg_away_odds': [], 'avg_btts_yes_prob': [],
            'leagues': set(), 'recent_form': [], 'surprise_results': 0,
            'bookmaker_accuracy': [], 'favorite_wins': 0, 'underdog_wins': 0
        })
        
        # Sort by start time to get chronological order for form calculation
        df_sorted = df.sort_values('start_time')
        
        for _, match in df_sorted.iterrows():
            home_team = match['home_team']
            away_team = match['away_team']
            league = match['league']
            
            # Extract actual scores if available
            if 'actual_home_score' in match and 'actual_away_score' in match:
                home_score = match['actual_home_score']
                away_score = match['actual_away_score']
                actual_winner = match.get('actual_winner', -1)
                
                # Update home team stats
                self.team_stats[home_team]['total_matches'] += 1
                self.team_stats[home_team]['home_matches'] += 1
                self.team_stats[home_team]['goals_scored'].append(home_score)
                self.team_stats[home_team]['goals_conceded'].append(away_score)
                self.team_stats[home_team]['total_goals'].append(home_score + away_score)
                self.team_stats[home_team]['leagues'].add(league)
                
                if actual_winner == 2:  # Home win
                    self.team_stats[home_team]['wins'] += 1
                    self.team_stats[home_team]['home_wins'] += 1
                    self.team_stats[home_team]['recent_form'].append('W')
                elif actual_winner == 1:  # Draw
                    self.team_stats[home_team]['draws'] += 1
                    self.team_stats[home_team]['home_draws'] += 1
                    self.team_stats[home_team]['recent_form'].append('D')
                else:  # Away win
                    self.team_stats[home_team]['losses'] += 1
                    self.team_stats[home_team]['home_losses'] += 1
                    self.team_stats[home_team]['recent_form'].append('L')
                
                # Update away team stats
                self.team_stats[away_team]['total_matches'] += 1
                self.team_stats[away_team]['away_matches'] += 1
                self.team_stats[away_team]['goals_scored'].append(away_score)
                self.team_stats[away_team]['goals_conceded'].append(home_score)
                self.team_stats[away_team]['total_goals'].append(home_score + away_score)
                self.team_stats[away_team]['leagues'].add(league)
                
                if actual_winner == 0:  # Away win
                    self.team_stats[away_team]['wins'] += 1
                    self.team_stats[away_team]['away_wins'] += 1
                    self.team_stats[away_team]['recent_form'].append('W')
                elif actual_winner == 1:  # Draw
                    self.team_stats[away_team]['draws'] += 1
                    self.team_stats[away_team]['away_draws'] += 1
                    self.team_stats[away_team]['recent_form'].append('D')
                else:  # Home win
                    self.team_stats[away_team]['losses'] += 1
                    self.team_stats[away_team]['away_losses'] += 1
                    self.team_stats[away_team]['recent_form'].append('L')
                
                # Analyze bookmaker accuracy and surprise results
                if all(col in match for col in ['home_win_odds', 'away_win_odds', 'draw_odds']):
                    odds = [match['away_win_odds'], match['draw_odds'], match['home_win_odds']]
                    favorite = np.argmin(odds)  # Team with lowest odds
                    
                    # Track favorite vs underdog performance
                    for team, is_home in [(home_team, True), (away_team, False)]:
                        team_outcome = actual_winner if is_home else (2 - actual_winner if actual_winner != 1 else 1)
                        expected_outcome = favorite if is_home else (2 - favorite if favorite != 1 else 1)
                        
                        if team_outcome == expected_outcome:
                            self.team_stats[team]['favorite_wins'] += 1
                        else:
                            self.team_stats[team]['underdog_wins'] += 1
                        
                        # Track surprise results (when heavy underdog wins)
                        if is_home and actual_winner == 2 and match['home_win_odds'] > 3.0:
                            self.team_stats[team]['surprise_results'] += 1
                        elif not is_home and actual_winner == 0 and match['away_win_odds'] > 3.0:
                            self.team_stats[team]['surprise_results'] += 1
            
            # Add odds data for both teams
            for team in [home_team, away_team]:
                if 'home_win_odds' in match:
                    self.team_stats[team]['avg_home_odds'].append(match['home_win_odds'])
                if 'away_win_odds' in match:
                    self.team_stats[team]['avg_away_odds'].append(match['away_win_odds'])
                if 'btts_yes_prob' in match:
                    self.team_stats[team]['avg_btts_yes_prob'].append(match['btts_yes_prob'])
        
        # Calculate final statistics
        for team in self.team_stats:
            stats = self.team_stats[team]
            
            # Win percentages
            if stats['total_matches'] > 0:
                stats['win_rate'] = stats['wins'] / stats['total_matches']
                stats['draw_rate'] = stats['draws'] / stats['total_matches']
                stats['loss_rate'] = stats['losses'] / stats['total_matches']
            
            if stats['home_matches'] > 0:
                stats['home_win_rate'] = stats['home_wins'] / stats['home_matches']
            
            if stats['away_matches'] > 0:
                stats['away_win_rate'] = stats['away_wins'] / stats['away_matches']
            
            # Goal statistics
            if stats['goals_scored']:
                stats['avg_goals_scored'] = np.mean(stats['goals_scored'])
                stats['avg_goals_conceded'] = np.mean(stats['goals_conceded'])
                stats['goal_difference'] = stats['avg_goals_scored'] - stats['avg_goals_conceded']
            
            # Recent form (last 5 matches)
            if stats['recent_form']:
                recent_5 = stats['recent_form'][-5:]
                stats['recent_wins'] = recent_5.count('W')
                stats['recent_form_points'] = recent_5.count('W') * 3 + recent_5.count('D')
                stats['form_trend'] = len(recent_5)  # Number of recent matches
            
            # Convert lists to averages
            for key in ['avg_home_odds', 'avg_away_odds', 'avg_btts_yes_prob']:
                if stats[key]:
                    stats[key] = np.mean(stats[key])
                else:
                    stats[key] = 2.0 if 'odds' in key else 0.5
            
            # Overall strength based on actual performance
            stats['performance_strength'] = (
                stats.get('win_rate', 0) * 0.4 +
                stats.get('goal_difference', 0) * 0.3 +
                stats.get('recent_form_points', 0) / 15 * 0.3  # Normalize recent form
            )
    
    def extract_features_from_match(self, match_data):
        """Extract comprehensive features from a single match (enhanced version)"""
        features = {}
        
        # Basic match info
        features['home_team'] = match_data['participants'][0]['name']
        features['away_team'] = match_data['participants'][1]['name']
        features['league'] = match_data['competition']['name']
        features['region'] = match_data['region']['name']
        features['match_id'] = match_data['id']
        features['start_time'] = match_data['startTime']
        
        # Parse start time for temporal features
        try:
            start_dt = datetime.fromisoformat(match_data['startTime'].replace('Z', '+00:00'))
            features['hour'] = start_dt.hour
            features['day_of_week'] = start_dt.weekday()
            features['month'] = start_dt.month
            features['is_weekend'] = 1 if start_dt.weekday() >= 5 else 0
            features['is_evening'] = 1 if start_dt.hour >= 18 else 0
        except:
            features['hour'] = 12
            features['day_of_week'] = 0
            features['month'] = 1
            features['is_weekend'] = 0
            features['is_evening'] = 0
        
        # Initialize market features with defaults
        market_features = {
            'home_win_odds': 2.0, 'home_win_prob': 0.33, 'home_win_value': 0,
            'home_refund_value': 0, 'home_key_value': 0,
            'away_win_odds': 2.0, 'away_win_prob': 0.33, 'away_win_value': 0,
            'away_refund_value': 0, 'away_key_value': 0,
            'draw_odds': 3.0, 'draw_prob': 0.33, 'draw_win_value': 0,
            'draw_refund_value': 0, 'draw_key_value': 0,
            'btts_yes_odds': 2.0, 'btts_yes_prob': 0.5, 'btts_yes_win_value': 0,
            'btts_no_odds': 2.0, 'btts_no_prob': 0.5, 'btts_no_win_value': 0
        }
        
        ou_lines = [1.5, 2.5, 3.5]
        for line in ou_lines:
            market_features.update({
                f'over_{line}_odds': 2.0, f'over_{line}_prob': 0.5, f'over_{line}_win_value': 0,
                f'under_{line}_odds': 2.0, f'under_{line}_prob': 0.5, f'under_{line}_win_value': 0,
            })
        
        # Extract odds and probabilities
        for market in match_data.get('markets', []):
            market_name = market['marketType']['name']
            
            if market_name == '1X2 - FT':
                for row in market['row']:
                    for price in row['prices']:
                        if price['name'] == '1':
                            prob_data = self.decode_probability(price['probability'])
                            market_features.update({
                                'home_win_odds': float(price['price']),
                                'home_win_prob': prob_data['probability'],
                                'home_win_value': prob_data['win_value'],
                                'home_refund_value': prob_data['refund_value'],
                                'home_key_value': prob_data['key_value']
                            })
                        elif price['name'] == 'X':
                            prob_data = self.decode_probability(price['probability'])
                            market_features.update({
                                'draw_odds': float(price['price']),
                                'draw_prob': prob_data['probability'],
                                'draw_win_value': prob_data['win_value'],
                                'draw_refund_value': prob_data['refund_value'],
                                'draw_key_value': prob_data['key_value']
                            })
                        elif price['name'] == '2':
                            prob_data = self.decode_probability(price['probability'])
                            market_features.update({
                                'away_win_odds': float(price['price']),
                                'away_win_prob': prob_data['probability'],
                                'away_win_value': prob_data['win_value'],
                                'away_refund_value': prob_data['refund_value'],
                                'away_key_value': prob_data['key_value']
                            })
            
            elif market_name == 'Total Score Over/Under - FT':
                for row in market['row']:
                    handicap = row.get('handicap', 0)
                    goal_line = handicap / 4
                    
                    if goal_line in ou_lines:
                        for price in row['prices']:
                            if price['name'] == 'Over':
                                prob_data = self.decode_probability(price['probability'])
                                market_features.update({
                                    f'over_{goal_line}_odds': float(price['price']),
                                    f'over_{goal_line}_prob': prob_data['probability'],
                                    f'over_{goal_line}_win_value': prob_data['win_value']
                                })
                            elif price['name'] == 'Under':
                                prob_data = self.decode_probability(price['probability'])
                                market_features.update({
                                    f'under_{goal_line}_odds': float(price['price']),
                                    f'under_{goal_line}_prob': prob_data['probability'],
                                    f'under_{goal_line}_win_value': prob_data['win_value']
                                })
            
            elif market_name == 'Both Teams To Score - FT':
                for row in market['row']:
                    for price in row['prices']:
                        if price['name'] == 'Yes':
                            prob_data = self.decode_probability(price['probability'])
                            market_features.update({
                                'btts_yes_odds': float(price['price']),
                                'btts_yes_prob': prob_data['probability'],
                                'btts_yes_win_value': prob_data['win_value']
                            })
                        elif price['name'] == 'No':
                            prob_data = self.decode_probability(price['probability'])
                            market_features.update({
                                'btts_no_odds': float(price['price']),
                                'btts_no_prob': prob_data['probability'],
                                'btts_no_win_value': prob_data['win_value']
                            })
        
        features.update(market_features)
        features.update(self.calculate_derived_features(features))
        
        return features
    
    def calculate_derived_features(self, features):
        """Calculate advanced derived features including result-based patterns"""
        derived = {}
        
        # Original odds-based features
        if all(k in features for k in ['home_win_odds', 'away_win_odds', 'draw_odds']):
            home_implied = 1 / features['home_win_odds']
            away_implied = 1 / features['away_win_odds']
            draw_implied = 1 / features['draw_odds']
            
            total_implied = home_implied + away_implied + draw_implied
            derived['bookmaker_margin'] = total_implied - 1 if total_implied > 0 else 0
            
            if total_implied > 0:
                derived['home_norm_prob'] = home_implied / total_implied
                derived['away_norm_prob'] = away_implied / total_implied
                derived['draw_norm_prob'] = draw_implied / total_implied
            else:
                derived['home_norm_prob'] = 0.33
                derived['away_norm_prob'] = 0.33
                derived['draw_norm_prob'] = 0.33
            
            # Enhanced odds analysis
            derived['home_away_odds_ratio'] = features['home_win_odds'] / features['away_win_odds'] if features['away_win_odds'] != 0 else 1.0
            derived['max_min_odds_spread'] = max(features['home_win_odds'], features['away_win_odds'], features['draw_odds']) - min(features['home_win_odds'], features['away_win_odds'], features['draw_odds'])
            
            # Market efficiency indicators
            min_odds = min(features['home_win_odds'], features['away_win_odds'], features['draw_odds'])
            max_odds = max(features['home_win_odds'], features['away_win_odds'], features['draw_odds'])
            derived['odds_efficiency'] = min_odds / max_odds  # Higher = more balanced market
            
            # Favorite identification with strength
            if features['home_win_odds'] == min_odds:
                derived['favorite'] = 2
                derived['favorite_strength'] = (features['away_win_odds'] + features['draw_odds']) / (2 * features['home_win_odds'])
            elif features['away_win_odds'] == min_odds:
                derived['favorite'] = 0
                derived['favorite_strength'] = (features['home_win_odds'] + features['draw_odds']) / (2 * features['away_win_odds'])
            else:
                derived['favorite'] = 1
                derived['favorite_strength'] = (features['home_win_odds'] + features['away_win_odds']) / (2 * features['draw_odds'])
        
        # Enhanced probability features
        if all(k in features for k in ['home_win_prob', 'away_win_prob', 'draw_prob']):
            probs = [features['home_win_prob'], features['away_win_prob'], features['draw_prob']]
            probs = [max(p, 1e-10) for p in probs]
            
            derived['prob_entropy'] = -sum(p * np.log(p) for p in probs)
            derived['prob_dominance'] = max(probs) - np.mean(probs)
            derived['prob_variance'] = np.var(probs)
            derived['prob_skewness'] = np.mean([(p - np.mean(probs))**3 for p in probs]) / (np.std(probs)**3 + 1e-10)
        
        # Enhanced Over/Under features
        ou_lines = [1.5, 2.5, 3.5]
        for line in ou_lines:
            over_key = f'over_{line}_prob'
            under_key = f'under_{line}_prob'
            
            if over_key in features and under_key in features:
                derived[f'ou_{line}_diff'] = features[over_key] - features[under_key]
                derived[f'ou_{line}_entropy'] = -(features[over_key] * np.log(max(features[over_key], 1e-10)) + 
                                                 features[under_key] * np.log(max(features[under_key], 1e-10)))
                derived[f'ou_{line}_confidence'] = abs(features[over_key] - features[under_key])
        
        # BTTS enhanced features
        if 'btts_yes_prob' in features and 'btts_no_prob' in features:
            derived['btts_diff'] = features['btts_yes_prob'] - features['btts_no_prob']
            derived['btts_entropy'] = -(features['btts_yes_prob'] * np.log(max(features['btts_yes_prob'], 1e-10)) + 
                                       features['btts_no_prob'] * np.log(max(features['btts_no_prob'], 1e-10)))
            derived['btts_confidence'] = abs(features['btts_yes_prob'] - features['btts_no_prob'])
        
        return derived
    
    def add_enhanced_team_features(self, df):
        """Add enhanced team features including actual performance metrics"""
        print("Adding enhanced team-based features...")
        
        for idx in df.index:
            home_team = df.loc[idx, 'home_team']
            away_team = df.loc[idx, 'away_team']
            
            # Enhanced home team features
            home_stats = self.team_stats.get(home_team, {})
            df.loc[idx, 'home_team_win_rate'] = home_stats.get('win_rate', 0.33)
            df.loc[idx, 'home_team_home_win_rate'] = home_stats.get('home_win_rate', 0.33)
            df.loc[idx, 'home_team_avg_goals_scored'] = home_stats.get('avg_goals_scored', 1.5)
            df.loc[idx, 'home_team_avg_goals_conceded'] = home_stats.get('avg_goals_conceded', 1.5)
            df.loc[idx, 'home_team_goal_diff'] = home_stats.get('goal_difference', 0)
            df.loc[idx, 'home_team_recent_form'] = home_stats.get('recent_form_points', 7.5)
            df.loc[idx, 'home_team_surprise_rate'] = home_stats.get('surprise_results', 0) / max(home_stats.get('total_matches', 1), 1)
            df.loc[idx, 'home_team_performance_strength'] = home_stats.get('performance_strength', 0.5)
            
            # Enhanced away team features
            away_stats = self.team_stats.get(away_team, {})
            df.loc[idx, 'away_team_win_rate'] = away_stats.get('win_rate', 0.33)
            df.loc[idx, 'away_team_away_win_rate'] = away_stats.get('away_win_rate', 0.33)
            df.loc[idx, 'away_team_avg_goals_scored'] = away_stats.get('avg_goals_scored', 1.5)
            df.loc[idx, 'away_team_avg_goals_conceded'] = away_stats.get('avg_goals_conceded', 1.5)
            df.loc[idx, 'away_team_goal_diff'] = away_stats.get('goal_difference', 0)
            df.loc[idx, 'away_team_recent_form'] = away_stats.get('recent_form_points', 7.5)
            df.loc[idx, 'away_team_surprise_rate'] = away_stats.get('surprise_results', 0) / max(away_stats.get('total_matches', 1), 1)
            df.loc[idx, 'away_team_performance_strength'] = away_stats.get('performance_strength', 0.5)
            
            # Enhanced comparison features
            df.loc[idx, 'performance_strength_diff'] = home_stats.get('performance_strength', 0.5) - away_stats.get('performance_strength', 0.5)
            df.loc[idx, 'goal_scoring_diff'] = home_stats.get('avg_goals_scored', 1.5) - away_stats.get('avg_goals_scored', 1.5)
            df.loc[idx, 'defensive_diff'] = away_stats.get('avg_goals_conceded', 1.5) - home_stats.get('avg_goals_conceded', 1.5)  # Lower conceded = better defense
            df.loc[idx, 'form_diff'] = home_stats.get('recent_form_points', 7.5) - away_stats.get('recent_form_points', 7.5)
            
            # Expected goals based on team averages
            df.loc[idx, 'expected_home_goals'] = (home_stats.get('avg_goals_scored', 1.5) + away_stats.get('avg_goals_conceded', 1.5)) / 2
            df.loc[idx, 'expected_away_goals'] = (away_stats.get('avg_goals_scored', 1.5) + home_stats.get('avg_goals_conceded', 1.5)) / 2
            df.loc[idx, 'expected_total_goals'] = df.loc[idx, 'expected_home_goals'] + df.loc[idx, 'expected_away_goals']
            
            # Head-to-head features (enhanced)
            h2h_key = f"{home_team}_vs_{away_team}"
            h2h_stats = self.h2h_stats.get(h2h_key, self.h2h_stats.get(f"{away_team}_vs_{home_team}", {}))
            
            df.loc[idx, 'h2h_matches'] = h2h_stats.get('matches', 0)
            df.loc[idx, 'h2h_avg_goals'] = h2h_stats.get('h2h_avg_goals', 2.5)
            df.loc[idx, 'h2h_home_favored_pct'] = h2h_stats.get('home_favored_pct', 0.5)
        
        return df
    
    def create_enhanced_targets(self, df):
        """Create target variables using actual match results"""
        targets = {}
        
        # Use actual results as ground truth when available
        if 'actual_winner' in df.columns:
            targets['winner'] = df['actual_winner'].values
            targets['winner_confident'] = np.ones(len(df), dtype=bool)  # All actual results are confident
            
            print(f"Using actual results for {len(df)} matches")
            
            # Analyze prediction vs actual for model improvement
            if all(col in df.columns for col in ['home_win_prob', 'away_win_prob', 'draw_prob']):
                predicted_probs = df[['away_win_prob', 'draw_prob', 'home_win_prob']].values
                predicted_winners = np.argmax(predicted_probs, axis=1)
                
                # Calculate bookmaker accuracy
                correct_predictions = (predicted_winners == targets['winner']).sum()
                bookmaker_accuracy = correct_predictions / len(df)
                print(f"Bookmaker prediction accuracy: {bookmaker_accuracy:.1%}")
                
                # Store accuracy for model calibration
                self.prediction_accuracy['bookmaker'].append(bookmaker_accuracy)
        else:
            # Fallback to probability-based targets if no actual results
            if all(col in df.columns for col in ['home_win_prob', 'away_win_prob', 'draw_prob']):
                probs = df[['away_win_prob', 'draw_prob', 'home_win_prob']].values
                max_probs = np.max(probs, axis=1)
                confident_mask = max_probs > 0.4
                
                targets['winner'] = np.argmax(probs, axis=1)
                targets['winner_confident'] = confident_mask
                print(f"Using probability-based targets for {confident_mask.sum()} confident predictions")
        
        # Over/Under targets using actual results
        for goal_line in [1.5, 2.5, 3.5]:
            actual_col = f'actual_over_{goal_line}'
            if actual_col in df.columns:
                targets[f'over_under_{goal_line}'] = df[actual_col].values
                targets[f'over_under_{goal_line}_confident'] = np.ones(len(df), dtype=bool)
            else:
                # Fallback to probability-based
                over_col = f'over_{goal_line}_prob'
                under_col = f'under_{goal_line}_prob'
                if over_col in df.columns and under_col in df.columns:
                    over_under_diff = df[over_col] - df[under_col]
                    targets[f'over_under_{goal_line}'] = (over_under_diff > 0.0).astype(int)
                    targets[f'over_under_{goal_line}_confident'] = (abs(over_under_diff) > 0.1)
        
        # BTTS targets using actual results
        if 'actual_btts' in df.columns:
            targets['btts'] = df['actual_btts'].values
            targets['btts_confident'] = np.ones(len(df), dtype=bool)
        else:
            # Fallback to probability-based
            if 'btts_yes_prob' in df.columns and 'btts_no_prob' in df.columns:
                btts_diff = df['btts_yes_prob'] - df['btts_no_prob']
                targets['btts'] = (btts_diff > 0.0).astype(int)
                targets['btts_confident'] = (abs(btts_diff) > 0.1)
        
        return targets
    
    def load_training_data(self):
        """Load and enhance all training JSON files (excluding upcoming data files)"""
        all_matches = []
        
        if not os.path.exists(self.train_folder):
            print(f"Training folder {self.train_folder} not found.")
            return pd.DataFrame()
        
        json_files = [f for f in os.listdir(self.train_folder) if f.endswith('.json')]
        
        if not json_files:
            print(f"No JSON files found in {self.train_folder}")
            return pd.DataFrame()
        
        print(f"Found {len(json_files)} training files to process...")
        
        for filename in json_files:
            filepath = os.path.join(self.train_folder, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    
                matches_in_file = data.get('items', [])
                
                for match in matches_in_file:
                    features = self.extract_features_from_match(match)
                    # Mark as historical training data
                    features['is_upcoming'] = False
                    all_matches.append(features)
                    
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        
        return pd.DataFrame(all_matches)
    
    def prepare_features(self, df, is_training=False):
        """Enhanced feature preparation with result-based improvements"""
        if df.empty:
            return df, []
        
        df_copy = df.copy()
        
        # Encode categorical variables
        categorical_cols = ['home_team', 'away_team', 'league', 'region']
        
        for col in categorical_cols:
            if col in df_copy.columns:
                df_copy[col] = df_copy[col].astype(str)
                if is_training:
                    unique_values = df_copy[col].unique()
                    self.encoders[col] = LabelEncoder()
                    self.encoders[col].fit(unique_values)
                    df_copy[f'{col}_encoded'] = self.encoders[col].transform(df_copy[col])
                else:
                    if col in self.encoders:
                        new_categories = set(df_copy[col].unique()) - set(self.encoders[col].classes_)
                        
                        if new_categories:
                            print(f"Warning: Unknown categories found in {col}: {new_categories}")
                            if len(self.encoders[col].classes_) > 0:
                                df_copy.loc[df_copy[col].isin(new_categories), f'{col}_encoded'] = 0
                                df_copy.loc[~df_copy[col].isin(new_categories), f'{col}_encoded'] = self.encoders[col].transform(df_copy.loc[~df_copy[col].isin(new_categories), col])
                            else:
                                df_copy[f'{col}_encoded'] = 0
                        else:
                            df_copy[f'{col}_encoded'] = self.encoders[col].transform(df_copy[col])
                    else:
                        print(f"Warning: Encoder for {col} not found during prediction. Assigning 0.")
                        df_copy[f'{col}_encoded'] = 0
            else:
                if f'{col}_encoded' not in df_copy.columns:
                    df_copy[f'{col}_encoded'] = 0
        
        # Select feature columns (exclude non-feature columns)
        exclude_cols = categorical_cols + ['match_id', 'start_time', 'is_upcoming'] + [
            col for col in df_copy.columns if col.startswith('actual_')
        ]
        
        if is_training:
            self.feature_columns = [col for col in df_copy.columns if col not in exclude_cols]
        
        # Handle missing values and scaling
        for col in self.feature_columns:
            if col in df_copy.columns:
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
                
                if is_training:
                    fill_value = df_copy[col].median()
                    df_copy[col] = df_copy[col].fillna(fill_value)
                    self.fill_values[col] = fill_value
                else:
                    fill_value = self.fill_values.get(col, df_copy[col].median() if not df_copy[col].empty else 0)
                    df_copy[col] = df_copy[col].fillna(fill_value)
            else:
                fill_value = self.fill_values.get(col, 0)
                df_copy[col] = fill_value
        
        # Remove low-variance features during training
        if is_training:
            valid_cols = []
            for col in self.feature_columns:
                if col in df_copy.columns:
                    if df_copy[col].var() > 1e-10:
                        missing_pct = df_copy[col].isna().sum() / len(df_copy)
                        if missing_pct < 0.8:
                            valid_cols.append(col)
                    else:
                        print(f"Warning: Feature '{col}' has zero variance and will be removed.")
            self.feature_columns = valid_cols
        
        # Prepare final feature matrix
        final_df_features = pd.DataFrame(index=df_copy.index)
        for col in self.feature_columns:
            if col in df_copy.columns:
                final_df_features[col] = df_copy[col]
            else:
                final_df_features[col] = self.fill_values.get(col, 0)
        
        # Scale features
        if is_training:
            self.scalers['main'] = StandardScaler()
            scaled_features = self.scalers['main'].fit_transform(final_df_features)
        else:
            if 'main' in self.scalers:
                scaled_features = self.scalers['main'].transform(final_df_features)
            else:
                print("Warning: Scaler not found for prediction. Features will not be scaled.")
                scaled_features = final_df_features.values
        
        scaled_df = pd.DataFrame(scaled_features, columns=self.feature_columns, index=df_copy.index)
        
        return scaled_df, self.feature_columns
    
    def train_models(self):
        """Enhanced training with actual results integration"""
        print("=== ENHANCED TRAINING WITH ACTUAL RESULTS ===")
        
        # Reset statistics
        self.team_stats = defaultdict(dict)
        self.h2h_stats = defaultdict(dict)
        self.fill_values = {}
        
        # Load training data (odds) - exclude upcoming data files
        print("Loading training data (odds)...")
        self.training_data = self.load_training_data()
        
        if self.training_data.empty:
            print("No training data available")
            return
        
        # Load results data
        print("Loading results data...")
        self.results_data = self.load_results_data()
        
        if self.results_data.empty:
            print("No results data available. Training with odds only...")
            self.matched_data = self.training_data.copy()
        else:
            # Match odds with results
            self.matched_data = self.match_odds_with_results()
            
            if self.matched_data.empty:
                print("No matches found between odds and results. Using odds only...")
                self.matched_data = self.training_data.copy()
        
        print(f"Training with {len(self.matched_data)} matches")
        
        # Build enhanced team statistics
        self.build_enhanced_team_statistics(self.matched_data.copy())
        
        # Add enhanced team features
        self.matched_data = self.add_enhanced_team_features(self.matched_data)
        
        # Prepare features
        X, self.feature_columns = self.prepare_features(self.matched_data.copy(), is_training=True)
        
        if X.empty or not self.feature_columns:
            print("No features available for training")
            return
        
        print(f"Using {len(self.feature_columns)} features for training")
        
        # Create enhanced targets
        targets = self.create_enhanced_targets(self.matched_data)
        
        if not targets:
            print("No targets available for training")
            return
        
        # Train models with enhanced approach
        for target_name, y in targets.items():
            if target_name.endswith('_confident'):
                continue
            
            print(f"\nTraining enhanced model for {target_name}...")
            
            # Use confident samples
            confident_mask_name = f'{target_name}_confident'
            if confident_mask_name in targets:
                confident_mask = targets[confident_mask_name]
                X_filtered = X[confident_mask]
                y_filtered = y[confident_mask]
                print(f"Using {sum(confident_mask)}/{len(confident_mask)} samples")
            else:
                X_filtered = X
                y_filtered = y
            
            if len(X_filtered) < 20 or len(np.unique(y_filtered)) < 2:
                print(f"Not enough diverse data for {target_name}, skipping...")
                continue
            
            # Enhanced model selection with hyperparameter tuning
            best_model = self.train_enhanced_model(X_filtered, y_filtered, target_name)
            
            if best_model:
                self.models[target_name] = best_model
                
                # Calculate and store accuracy metrics
                y_pred = best_model.predict(X_filtered)
                accuracy = accuracy_score(y_filtered, y_pred)
                self.prediction_accuracy[target_name].append(accuracy)
                
                print(f"Model trained for {target_name} with {accuracy:.1%} accuracy")
    
    def train_enhanced_model(self, X, y, target_name):
        """Train model with enhanced algorithms and hyperparameter tuning"""
        # Split data
        y_series = pd.Series(y)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_series, test_size=0.2, random_state=42, stratify=y_series
        )
        
        # Enhanced model configurations
        models_to_test = {}
        
        # XGBoost with enhanced parameters
        models_to_test['xgboost'] = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        )
        
        # Random Forest with enhanced parameters
        models_to_test['random_forest'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=3,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            class_weight='balanced'
        )
        
        # Enhanced Decision Tree
        models_to_test['decision_tree'] = DecisionTreeClassifier(
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            class_weight='balanced'
        )
        
        # Logistic Regression with regularization
        models_to_test['logistic'] = LogisticRegression(
            max_iter=2000,
            random_state=42,
            class_weight='balanced',
            C=0.1,  # Regularization
            solver='liblinear'
        )
        
        # Train and evaluate models
        best_model = None
        best_score = -1
        best_name = ""
        
        for model_name, model in models_to_test.items():
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Cross-validation with stratification
                cv_scores = cross_val_score(model, X_train, y_train, cv=min(5, len(X_train)//10), scoring='accuracy')
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
                
                # Test set evaluation
                test_score = model.score(X_test, y_test)
                
                print(f"  {model_name}:")
                print(f"    CV: {cv_mean:.3f} ± {cv_std:.3f}")
                print(f"    Test: {test_score:.3f}")
                
                # Select best model (prioritize CV score with stability)
                stability_score = cv_mean - cv_std  # Penalize high variance
                if stability_score > best_score:
                    best_score = stability_score
                    best_model = model
                    best_name = model_name
            
            except Exception as e:
                print(f"    Error training {model_name}: {e}")
        
        if best_model:
            print(f"  Selected: {best_name} (Stability Score: {best_score:.3f})")
            
            # Feature importance analysis
            if hasattr(best_model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': self.feature_columns,
                    'importance': best_model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                print(f"  Top 5 features:")
                for _, row in importance_df.head(5).iterrows():
                    print(f"    {row['feature']}: {row['importance']:.3f}")
        
        return best_model
    
    def evaluate_prediction_vs_actual(self, prediction_data, actual_data):
        """Evaluate how well our predictions match actual results"""
        if not self.models or actual_data.empty:
            print("No models or actual data available for evaluation")
            return
        
        print("\n=== PREDICTION VS ACTUAL EVALUATION ===")
        
        # Match predictions with actual results
        matched_predictions = []
        
        for _, pred_match in prediction_data.iterrows():
            # Find corresponding actual result
            actual_match = actual_data[
                (actual_data['home_team'] == pred_match['home_team']) &
                (actual_data['away_team'] == pred_match['away_team']) &
                (abs(pd.to_datetime(actual_data['start_time']) - 
                     pd.to_datetime(pred_match['start_time'])).dt.total_seconds() <= 600)
            ]
            
            if not actual_match.empty:
                actual_result = actual_match.iloc[0]
                
                # Prepare prediction features
                pred_df = pd.DataFrame([pred_match])
                pred_df = self.add_enhanced_team_features(pred_df)
                X_pred, _ = self.prepare_features(pred_df, is_training=False)
                
                if not X_pred.empty:
                    # Get model predictions
                    predictions = {}
                    for target_name, model in self.models.items():
                        try:
                            pred = model.predict(X_pred)[0]
                            if hasattr(model, 'predict_proba'):
                                pred_proba = model.predict_proba(X_pred)[0]
                                confidence = max(pred_proba)
                            else:
                                pred_proba = None
                                confidence = 0.5
                            
                            predictions[target_name] = {
                                'prediction': pred,
                                'confidence': confidence,
                                'probabilities': pred_proba
                            }
                        except Exception as e:
                            print(f"Error predicting {target_name}: {e}")
                    
                    # Compare with actual results
                    evaluation = {
                        'match': f"{pred_match['home_team']} vs {pred_match['away_team']}",
                        'league': pred_match['league'],
                        'predictions': predictions,
                        'actual_results': {
                            'winner': actual_result.get('actual_winner', -1),
                            'total_goals': actual_result.get('actual_total_goals', 0),
                            'btts': actual_result.get('actual_btts', 0)
                        }
                    }
                    
                    matched_predictions.append(evaluation)
        
        # Calculate overall accuracy
        if matched_predictions:
            print(f"Evaluated {len(matched_predictions)} predictions against actual results:")
            
            for target in ['winner', 'btts']:
                if target in self.models:
                    correct = 0
                    total = 0
                    
                    for eval_data in matched_predictions:
                        if target in eval_data['predictions'] and target in eval_data['actual_results']:
                            predicted = eval_data['predictions'][target]['prediction']
                            actual = eval_data['actual_results'][target]
                            
                            if actual != -1:  # Valid actual result
                                total += 1
                                if predicted == actual:
                                    correct += 1
                    
                    if total > 0:
                        accuracy = correct / total
                        print(f"  {target}: {accuracy:.1%} ({correct}/{total})")
                        self.prediction_accuracy[f'{target}_vs_actual'].append(accuracy)
        
        return matched_predictions
    
    def predict_match(self, home_team, away_team, league):
        """Enhanced prediction with result-based learning (original method)"""
        if not self.models:
            print("No trained models available. Please train first.")
            return None
        
        # Create match data template
        now_dt = datetime.now()
        market_features_template = {
            'home_win_odds': 2.0, 'home_win_prob': 0.33, 'home_win_value': 0,
            'home_refund_value': 0, 'home_key_value': 0,
            'away_win_odds': 2.0, 'away_win_prob': 0.33, 'away_win_value': 0,
            'away_refund_value': 0, 'away_key_value': 0,
            'draw_odds': 3.0, 'draw_prob': 0.33, 'draw_win_value': 0,
            'draw_refund_value': 0, 'draw_key_value': 0,
            'btts_yes_odds': 2.0, 'btts_yes_prob': 0.5, 'btts_yes_win_value': 0,
            'btts_no_odds': 2.0, 'btts_no_prob': 0.5, 'btts_no_win_value': 0
        }
        
        ou_lines = [1.5, 2.5, 3.5]
        for line in ou_lines:
            market_features_template.update({
                f'over_{line}_odds': 2.0, f'over_{line}_prob': 0.5, f'over_{line}_win_value': 0,
                f'under_{line}_odds': 2.0, f'under_{line}_prob': 0.5, f'under_{line}_win_value': 0,
            })
        
        match_data = {
            'home_team': home_team,
            'away_team': away_team,
            'league': league,
            'region': 'Unknown',
            'match_id': 'PREDICTED_MATCH',
            'start_time': now_dt.isoformat(),
            'hour': now_dt.hour,
            'day_of_week': now_dt.weekday(),
            'month': now_dt.month,
            'is_weekend': 1 if now_dt.weekday() >= 5 else 0,
            'is_evening': 1 if now_dt.hour >= 18 else 0,
            'is_upcoming': False,
            **market_features_template
        }
        
        # Enhanced odds estimation using historical performance
        similar_matches = self.matched_data[
            ((self.matched_data['home_team'] == home_team) | 
             (self.matched_data['away_team'] == away_team)) &
            (self.matched_data['league'] == league)
        ]
        
        if not similar_matches.empty:
            for col in market_features_template.keys():
                if col in similar_matches.columns and not similar_matches[col].isna().all():
                    # Weight recent matches more heavily
                    if 'start_time' in similar_matches.columns:
                        similar_matches_sorted = similar_matches.sort_values('start_time', ascending=False)
                        recent_matches = similar_matches_sorted.head(10)  # Last 10 matches
                        match_data[col] = recent_matches[col].mean()
                    else:
                        match_data[col] = similar_matches[col].mean()
        
        # Add derived features
        derived = self.calculate_derived_features(match_data)
        match_data.update(derived)
        
        # Convert to DataFrame and add team features
        match_df = pd.DataFrame([match_data])
        match_df = self.add_enhanced_team_features(match_df)
        
        # Prepare features for prediction
        X, _ = self.prepare_features(match_df, is_training=False)
        
        predictions = {}
        probabilities = {}
        confidence_scores = {}
        
        for target_name, model in self.models.items():
            try:
                pred = model.predict(X)[0]
                
                if hasattr(model, 'predict_proba'):
                    pred_proba = model.predict_proba(X)[0]
                    confidence = max(pred_proba)
                    
                    # Adjust confidence based on historical accuracy
                    if target_name in self.prediction_accuracy:
                        historical_accuracy = np.mean(self.prediction_accuracy[target_name])
                        adjusted_confidence = confidence * historical_accuracy
                    else:
                        adjusted_confidence = confidence
                else:
                    pred_proba = None
                    confidence = 0.5
                    adjusted_confidence = 0.5
                
                predictions[target_name] = pred
                probabilities[target_name] = pred_proba
                confidence_scores[target_name] = adjusted_confidence
                
            except Exception as e:
                print(f"Error predicting {target_name}: {e}")
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'confidence': confidence_scores,
            'match_info': {
                'home_team': home_team,
                'away_team': away_team,
                'league': league,
                'using_current_odds': False
            },
            'enhanced_analysis': self.get_enhanced_match_analysis(home_team, away_team, league)
        }
    
    def get_enhanced_match_analysis(self, home_team, away_team, league):
        """Get enhanced analysis based on actual performance data"""
        analysis = {}
        
        home_stats = self.team_stats.get(home_team, {})
        away_stats = self.team_stats.get(away_team, {})
        
        # Performance-based analysis
        analysis['home_recent_form'] = home_stats.get('recent_form_points', 7.5)
        analysis['away_recent_form'] = away_stats.get('recent_form_points', 7.5)
        analysis['home_goal_avg'] = home_stats.get('avg_goals_scored', 1.5)
        analysis['away_goal_avg'] = away_stats.get('avg_goals_scored', 1.5)
        analysis['home_defense'] = home_stats.get('avg_goals_conceded', 1.5)
        analysis['away_defense'] = away_stats.get('avg_goals_conceded', 1.5)
        
        # Expected match outcome based on team performance
        home_strength = home_stats.get('performance_strength', 0.5)
        away_strength = away_stats.get('performance_strength', 0.5)
        
        analysis['expected_winner'] = 'Home' if home_strength > away_strength else 'Away'
        analysis['strength_difference'] = abs(home_strength - away_strength)
        analysis['match_competitiveness'] = 1 - analysis['strength_difference']  # Higher = more competitive
        
        # Goal expectation
        expected_home_goals = (home_stats.get('avg_goals_scored', 1.5) + away_stats.get('avg_goals_conceded', 1.5)) / 2
        expected_away_goals = (away_stats.get('avg_goals_scored', 1.5) + home_stats.get('avg_goals_conceded', 1.5)) / 2
        
        analysis['expected_total_goals'] = expected_home_goals + expected_away_goals
        analysis['expected_btts'] = 'Yes' if expected_home_goals > 0.8 and expected_away_goals > 0.8 else 'No'
        
        return analysis
    
    def predict_with_enhanced_reasoning(self, home_team, away_team, league, upcoming_filename=None):
        """Provide enhanced predictions with detailed reasoning based on actual performance"""
        if upcoming_filename:
            result = self.predict_with_upcoming_data(home_team, away_team, league, upcoming_filename)
        else:
            result = self.predict_match(home_team, away_team, league)
        
        if not result:
            return None
        
        print(f"\n=== ENHANCED PREDICTION ANALYSIS: {home_team} vs {away_team} ===")
        
        # Show if using current odds
        if result['match_info'].get('using_current_odds', False):
            print(f"🎯 USING CURRENT ODDS DATA from {result['current_odds_data']['source_file']}")
            print(f"Current Odds: Home {result['current_odds_data']['home_win_odds']:.2f} | "
                  f"Draw {result['current_odds_data']['draw_odds']:.2f} | "
                  f"Away {result['current_odds_data']['away_win_odds']:.2f}")
        else:
            print("📊 Using historical data prediction (no current odds provided)")
        
        # Enhanced team analysis
        enhanced_analysis = result['enhanced_analysis']
        
        print(f"\nPERFORMANCE-BASED TEAM ANALYSIS:")
        print(f"  {home_team}:")
        print(f"    Recent form: {enhanced_analysis['home_recent_form']:.1f}/15 points")
        print(f"    Goal average: {enhanced_analysis['home_goal_avg']:.2f}")
        print(f"    Defense: {enhanced_analysis['home_defense']:.2f} goals conceded")
        
        print(f"  {away_team}:")
        print(f"    Recent form: {enhanced_analysis['away_recent_form']:.1f}/15 points")
        print(f"    Goal average: {enhanced_analysis['away_goal_avg']:.2f}")
        print(f"    Defense: {enhanced_analysis['away_defense']:.2f} goals conceded")
        
        print(f"\nMATCH COMPETITIVENESS: {enhanced_analysis['match_competitiveness']:.1%}")
        print(f"EXPECTED TOTAL GOALS: {enhanced_analysis['expected_total_goals']:.2f}")
        
        # Enhanced predictions with historical accuracy
        predictions = result['predictions']
        confidence_scores = result['confidence']
        
        if 'winner' in predictions:
            winner_pred = predictions['winner']
            winner_map = {0: away_team, 1: 'Draw', 2: home_team}
            predicted_winner = winner_map.get(winner_pred, 'Unknown')
            confidence = confidence_scores['winner']
            
            print(f"\nENHANCED MATCH RESULT PREDICTION:")
            print(f"  Predicted Winner: {predicted_winner}")
            print(f"  Model Confidence: {confidence:.1%}")
            
            # Show odds analysis if available
            if 'odds_analysis' in result and 'winner' in result['odds_analysis']:
                odds_info = result['odds_analysis']['winner']
                print(f"  Market Favorite: {odds_info.get('market_favorite', 'Unknown')}")
                if odds_info.get('is_value_bet', False):
                    print(f"  🎯 VALUE BET DETECTED - Model disagrees with market!")
            
            # Historical accuracy context
            if 'winner' in self.prediction_accuracy:
                historical_acc = np.mean(self.prediction_accuracy['winner'])
                print(f"  Historical Accuracy: {historical_acc:.1%}")
            
            # Enhanced recommendation system
            if confidence > 0.80 and enhanced_analysis['strength_difference'] > 0.2:
                print(f"  Recommendation: STRONG BET - High confidence with clear strength advantage")
            elif confidence > 0.70:
                print(f"  Recommendation: GOOD BET - Solid confidence")
            elif confidence > 0.55:
                print(f"  Recommendation: CONSIDER - Moderate confidence")
            else:
                print(f"  Recommendation: AVOID - Low confidence or competitive match")
        
        # Enhanced goals prediction
        print(f"\nENHANCED GOALS ANALYSIS:")
        for target in predictions:
            if target.startswith('over_under'):
                goal_line = target.split('_')[-1]
                pred_val = predictions[target]
                confidence = confidence_scores[target]
                ou_map = {0: 'Under', 1: 'Over'}
                
                expected_goals = enhanced_analysis['expected_total_goals']
                line_value = float(goal_line)
                
                print(f"  {goal_line} goals: {ou_map.get(pred_val, 'Unknown')} (Confidence: {confidence:.1%})")
                print(f"    Expected goals: {expected_goals:.2f} vs Line: {line_value}")
                
                # Show odds analysis if available
                if 'odds_analysis' in result and target in result['odds_analysis']:
                    odds_info = result['odds_analysis'][target]
                    if odds_info.get('is_value_bet', False):
                        print(f"    🎯 VALUE BET OPPORTUNITY")
                
                if abs(expected_goals - line_value) > 0.5:
                    print(f"    Strong indicator: Expected goals significantly {'above' if expected_goals > line_value else 'below'} line")
        
        # Enhanced BTTS prediction
        if 'btts' in predictions:
            btts_pred = predictions['btts']
            btts_result = 'Yes' if btts_pred == 1 else 'No'
            btts_confidence = confidence_scores['btts']
            
            print(f"\nBOTH TEAMS TO SCORE:")
            print(f"  Prediction: {btts_result}")
            print(f"  Confidence: {btts_confidence:.1%}")
            print(f"  Expected: {enhanced_analysis['expected_btts']} (based on goal averages)")
            
            # Show odds analysis if available
            if 'odds_analysis' in result and 'btts' in result['odds_analysis']:
                odds_info = result['odds_analysis']['btts']
                if odds_info.get('is_value_bet', False):
                    print(f"  🎯 VALUE BET OPPORTUNITY")
        
        return result
    
    def continuous_learning_update(self, new_results_folder=None):
        """Update models with new results for continuous learning"""
        if new_results_folder:
            self.results_folder = new_results_folder
        
        print("=== CONTINUOUS LEARNING UPDATE ===")
        
        # Load new results
        new_results = self.load_results_data()
        
        if new_results.empty:
            print("No new results to learn from")
            return
        
        # Combine with existing results
        if not self.results_data.empty:
            combined_results = pd.concat([self.results_data, new_results], ignore_index=True)
            # Remove duplicates based on match_id and start_time
            combined_results = combined_results.drop_duplicates(subset=['match_id', 'start_time'])
        else:
            combined_results = new_results
        
        self.results_data = combined_results
        
        print(f"Updated with {len(new_results)} new results")
        print(f"Total results available: {len(self.results_data)}")
        
        # Retrain models with updated data
        print("Retraining models with updated results...")
        self.train_models()
    
    def get_model_performance_report(self):
        """Generate comprehensive model performance report"""
        if not self.prediction_accuracy:
            return "No performance data available"
        
        report = []
        report.append("=== ENHANCED MODEL PERFORMANCE REPORT ===")
        
        for target, accuracies in self.prediction_accuracy.items():
            if accuracies:
                avg_accuracy = np.mean(accuracies)
                latest_accuracy = accuracies[-1]
                improvement = latest_accuracy - accuracies[0] if len(accuracies) > 1 else 0
                
                report.append(f"\n{target.upper()}:")
                report.append(f"  Average Accuracy: {avg_accuracy:.1%}")
                report.append(f"  Latest Accuracy: {latest_accuracy:.1%}")
                report.append(f"  Improvement: {improvement:+.1%}")
                report.append(f"  Evaluations: {len(accuracies)}")
        
        # Team statistics summary
        if self.team_stats:
            report.append(f"\nTEAM STATISTICS:")
            report.append(f"  Teams analyzed: {len(self.team_stats)}")
            
            # Find best performing teams
            performance_teams = [(team, stats.get('performance_strength', 0)) 
                               for team, stats in self.team_stats.items()]
            performance_teams.sort(key=lambda x: x[1], reverse=True)
            
            report.append(f"  Top performing teams:")
            for team, strength in performance_teams[:5]:
                report.append(f"    {team}: {strength:.3f}")
        
        return "\n".join(report)
    
    def save_enhanced_models(self, filename='enhanced_sports_models_v2.pkl'):
        """Save enhanced models with all data"""
        model_data = {
            'models': self.models,
            'encoders': self.encoders,
            'scalers': self.scalers,
            'feature_columns': self.feature_columns,
            'training_data': self.training_data,
            'results_data': self.results_data,
            'matched_data': self.matched_data,
            'team_stats': dict(self.team_stats),
            'h2h_stats': dict(self.h2h_stats),
            'fill_values': self.fill_values,
            'prediction_accuracy': dict(self.prediction_accuracy)
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Enhanced models saved to {filename}")
    
    def load_enhanced_models(self, filename='enhanced_sports_models_v2.pkl'):
        """Load enhanced models with all data"""
        try:
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)
            
            self.models = model_data.get('models', {})
            self.encoders = model_data.get('encoders', {})
            self.scalers = model_data.get('scalers', {})
            self.feature_columns = model_data.get('feature_columns', [])
            self.training_data = model_data.get('training_data', pd.DataFrame())
            self.results_data = model_data.get('results_data', pd.DataFrame())
            self.matched_data = model_data.get('matched_data', pd.DataFrame())
            self.team_stats = defaultdict(dict, model_data.get('team_stats', {}))
            self.h2h_stats = defaultdict(dict, model_data.get('h2h_stats', {}))
            self.fill_values = model_data.get('fill_values', {})
            self.prediction_accuracy = defaultdict(list, model_data.get('prediction_accuracy', {}))
            
            print(f"Enhanced models loaded from {filename}")
            print(f"Loaded {len(self.models)} models")
            print(f"Team statistics for {len(self.team_stats)} teams")
            print(f"Results data: {len(self.results_data)} matches")
            print(f"Matched data: {len(self.matched_data)} matches")
            return True
        except FileNotFoundError:
            print(f"Model file {filename} not found")
            return False
        except Exception as e:
            print(f"Error loading models: {e}")
            return False

def main():
    """Enhanced main execution with upcoming data integration"""
    predictor = EnhancedSportsPredictor()
    
    # Try to load existing enhanced models
    if not predictor.load_enhanced_models():
        print("No existing enhanced models found. Training new models...")
        predictor.train_models()
        if predictor.models:
            predictor.save_enhanced_models()
    
    # Show performance report
    print(predictor.get_model_performance_report())
    
    # Enhanced interactive commands
    print("\n=== ENHANCED INTERACTIVE PREDICTION SYSTEM ===")
    print("Available commands:")
    print("1. predict <home_team> <away_team> <league> [upcoming_file.json] - Enhanced prediction")
    print("   Example: predict FRE FCB German League 30.json")
    print("   Example: predict FRE FCB German League (without upcoming data)")
    print("2. predict from <upcoming_file.json> - Predict all matches in upcoming file")
    print("   Example: predict from 30.json")
    print("3. evaluate - Evaluate predictions against actual results")
    print("4. update_results [folder] - Update with new results for continuous learning")
    print("5. retrain - Retrain models with all available data")
    print("6. performance - Show detailed performance report")
    print("7. team_analysis <team_name> - Show comprehensive team analysis")
    print("8. match_history <home_team> <away_team> - Show head-to-head history")
    print("9. league_patterns <league_name> - Analyze league-specific patterns")
    print("10. save - Save current models")
    print("11. quit")
    
    while True:
        try:
            command = input("\nEnter command: ").strip()
            
            if not command:
                continue
            
            command_parts = command.split()
            
            if command_parts[0] == 'quit':
                break
            
            elif command_parts[0] == 'predict':
                if len(command_parts) >= 2 and command_parts[1] == 'from':
                    # Predict from upcoming file
                    if len(command_parts) >= 3:
                        upcoming_filename = command_parts[2]
                        predictions = predictor.predict_all_upcoming_matches(upcoming_filename)
                        if predictions:
                            predictor.display_upcoming_predictions_summary(predictions)
                    else:
                        print("Please specify the upcoming data file: predict from <filename.json>")
                
                elif len(command_parts) >= 4:
                    # Individual match prediction
                    home_team = command_parts[1]
                    away_team = command_parts[2]
                    league = command_parts[3]
                    upcoming_filename = command_parts[4] if len(command_parts) >= 5 else None
                    
                    predictor.predict_with_enhanced_reasoning(home_team, away_team, league, upcoming_filename)
                else:
                    print("Usage: predict <home_team> <away_team> <league> [upcoming_file.json]")
                    print("   or: predict from <upcoming_file.json>")
                
            elif command_parts[0] == 'evaluate':
                if not predictor.training_data.empty and not predictor.results_data.empty:
                    predictor.evaluate_prediction_vs_actual(predictor.training_data, predictor.results_data)
                else:
                    print("Need both training data and results data for evaluation")
                
            elif command_parts[0] == 'update_results':
                folder = command_parts[1] if len(command_parts) > 1 else None
                predictor.continuous_learning_update(folder)
                
            elif command_parts[0] == 'retrain':
                print("Retraining enhanced models...")
                predictor.train_models()
                if predictor.models:
                    predictor.save_enhanced_models()
                
            elif command_parts[0] == 'performance':
                print(predictor.get_model_performance_report())
                
            elif command_parts[0] == 'team_analysis' and len(command_parts) >= 2:
                team_name = ' '.join(command_parts[1:])
                if team_name in predictor.team_stats:
                    stats = predictor.team_stats[team_name]
                    print(f"\n=== {team_name} COMPREHENSIVE ANALYSIS ===")
                    print(f"Performance Strength: {stats.get('performance_strength', 0):.3f}")
                    print(f"Total Matches: {stats.get('total_matches', 0)}")
                    print(f"Win Rate: {stats.get('win_rate', 0):.1%}")
                    print(f"Home Win Rate: {stats.get('home_win_rate', 0):.1%}")
                    print(f"Away Win Rate: {stats.get('away_win_rate', 0):.1%}")
                    print(f"Average Goals Scored: {stats.get('avg_goals_scored', 0):.2f}")
                    print(f"Average Goals Conceded: {stats.get('avg_goals_conceded', 0):.2f}")
                    print(f"Goal Difference: {stats.get('goal_difference', 0):.2f}")
                    print(f"Recent Form Points: {stats.get('recent_form_points', 0):.1f}/15")
                    print(f"Surprise Results: {stats.get('surprise_results', 0)}")
                    print(f"Leagues: {', '.join(stats.get('leagues', set()))}")
                else:
                    print(f"No data found for team: {team_name}")
                    
            elif command_parts[0] == 'save':
                predictor.save_enhanced_models()
                print("Models saved successfully")
                
            else:
                print("Invalid command. Type a valid command or 'quit' to exit.")
                
        except KeyboardInterrupt:
            print("\nExiting enhanced predictor.")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()