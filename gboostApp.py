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
from datetime import datetime
import warnings
from collections import defaultdict
warnings.filterwarnings('ignore')

class EnhancedSportsPredictor:
    def __init__(self, train_folder='train'):
        self.train_folder = train_folder
        self.models = {}
        self.encoders = {}
        self.scalers = {}
        self.feature_columns = []
        self.training_data = pd.DataFrame()
        self.team_stats = defaultdict(dict)
        self.h2h_stats = defaultdict(dict)
        self.fill_values = {} # To store fill values for prediction
        
    def decode_probability(self, prob_string):
        """Decode the base64 encoded probability string"""
        try:
            decoded = base64.b64decode(prob_string)
            prob_data = json.loads(decoded.decode('utf-8'))
            
            win_value = prob_data.get('win', 0)
            refund_value = prob_data.get('refund', 0)
            key_value = prob_data.get('key', 0)
            
            # Better probability calculation
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
            # print(f"Error decoding probability string {prob_string}: {e}") # Debugging
            return {
                'probability': 0.5,
                'win_value': 0,
                'refund_value': 0,
                'key_value': 0
            }
    
    def extract_features_from_match(self, match_data):
        """Extract comprehensive features from a single match"""
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
        except:
            features['hour'] = 12
            features['day_of_week'] = 0
            features['month'] = 1
        
        # Initialize all market features with defaults
        # This comprehensive initialization is crucial for consistent feature sets
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
        ou_lines = [1.5, 2.5, 3.5] # Consistent with calculate_derived_features
        for line in ou_lines:
            market_features.update({
                f'over_{line}_odds': 2.0, f'over_{line}_prob': 0.5, f'over_{line}_win_value': 0,
                f'under_{line}_odds': 2.0, f'under_{line}_prob': 0.5, f'under_{line}_win_value': 0,
            })
        
        # Extract odds and probabilities for different markets
        for market in match_data['markets']:
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
                    goal_line = handicap / 4  # Convert to goal line (assuming this is how goal lines are derived from handicap in source)
                    
                    # Only update for known goal lines (1.5, 2.5, 3.5)
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
        
        # Add market features to main features
        features.update(market_features)
        
        # Calculate advanced derived features
        features.update(self.calculate_derived_features(features))
        
        return features
    
    def calculate_derived_features(self, features):
        """Calculate advanced derived features for better prediction"""
        derived = {}
        
        # Odds-based features
        if all(k in features for k in ['home_win_odds', 'away_win_odds', 'draw_odds']):
            # Implied probabilities from odds
            home_implied = 1 / features['home_win_odds']
            away_implied = 1 / features['away_win_odds']
            draw_implied = 1 / features['draw_odds']
            
            # Bookmaker margin
            total_implied = home_implied + away_implied + draw_implied
            derived['bookmaker_margin'] = total_implied - 1 if total_implied > 0 else 0
            
            # Normalized probabilities
            if total_implied > 0:
                derived['home_norm_prob'] = home_implied / total_implied
                derived['away_norm_prob'] = away_implied / total_implied
                derived['draw_norm_prob'] = draw_implied / total_implied
            else:
                derived['home_norm_prob'] = 0.33
                derived['away_norm_prob'] = 0.33
                derived['draw_norm_prob'] = 0.33
            
            # Odds ratios and differences (handle division by zero)
            derived['home_away_odds_ratio'] = features['home_win_odds'] / features['away_win_odds'] if features['away_win_odds'] != 0 else 1.0
            derived['home_draw_odds_ratio'] = features['home_win_odds'] / features['draw_odds'] if features['draw_odds'] != 0 else 1.0
            derived['away_draw_odds_ratio'] = features['away_win_odds'] / features['draw_odds'] if features['draw_odds'] != 0 else 1.0
            
            # Odds spreads
            derived['max_min_odds_spread'] = max(features['home_win_odds'], features['away_win_odds'], features['draw_odds']) - min(features['home_win_odds'], features['away_win_odds'], features['draw_odds'])
            
            # Favorite identification
            min_odds = min(features['home_win_odds'], features['away_win_odds'], features['draw_odds'])
            if features['home_win_odds'] == min_odds:
                derived['favorite'] = 2  # Home favorite
            elif features['away_win_odds'] == min_odds:
                derived['favorite'] = 0  # Away favorite
            else:
                derived['favorite'] = 1  # Draw favorite (rare)
            
            # Underdog odds
            derived['underdog_odds'] = max(features['home_win_odds'], features['away_win_odds'])
            derived['favorite_odds'] = min(features['home_win_odds'], features['away_win_odds'])
            
            # Match competitiveness (closer odds = more competitive)
            derived['competitiveness'] = 1 / (abs(features['home_win_odds'] - features['away_win_odds']) + 1)
        
        # Probability-based features
        if all(k in features for k in ['home_win_prob', 'away_win_prob', 'draw_prob']):
            # Probability entropy (uncertainty measure)
            probs = [features['home_win_prob'], features['away_win_prob'], features['draw_prob']]
            probs = [max(p, 1e-10) for p in probs]  # Avoid log(0)
            derived['prob_entropy'] = -sum(p * np.log(p) for p in probs)
            
            # Probability dominance
            max_prob = max(probs)
            derived['prob_dominance'] = max_prob - np.mean(probs)
            
            # Probability variance
            derived['prob_variance'] = np.var(probs)
        
        # Over/Under features
        ou_lines = [1.5, 2.5, 3.5]
        for line in ou_lines:
            over_key = f'over_{line}_prob'
            under_key = f'under_{line}_prob'
            
            if over_key in features and under_key in features:
                derived[f'ou_{line}_diff'] = features[over_key] - features[under_key]
                derived[f'ou_{line}_entropy'] = -(features[over_key] * np.log(max(features[over_key], 1e-10)) + 
                                                 features[under_key] * np.log(max(features[under_key], 1e-10)))
        
        # BTTS features
        if 'btts_yes_prob' in features and 'btts_no_prob' in features:
            derived['btts_diff'] = features['btts_yes_prob'] - features['btts_no_prob']
            derived['btts_entropy'] = -(features['btts_yes_prob'] * np.log(max(features['btts_yes_prob'], 1e-10)) + 
                                       features['btts_no_prob'] * np.log(max(features['btts_no_prob'], 1e-10)))
        
        # Value-based features (from win/refund/key values)
        value_keys = ['home', 'away', 'draw']
        for key in value_keys:
            win_key = f'{key}_win_value'
            refund_key = f'{key}_refund_value'
            key_value_key = f'{key}_key_value'
            
            if all(k in features for k in [win_key, refund_key, key_value_key]):
                total_value = abs(features[win_key]) + abs(features[refund_key]) + abs(features[key_value_key]) + 1e-10
                derived[f'{key}_value_ratio'] = abs(features[win_key]) / total_value
                derived[f'{key}_refund_ratio'] = abs(features[refund_key]) / total_value
        
        return derived
    
    def build_team_statistics(self, df):
        """Build comprehensive team statistics from historical data"""
        print("Building team statistics...")
        
        # Ensure that avg_home_odds etc. are lists when starting
        # This is handled by initializing self.team_stats in train_models
        
        for _, match in df.iterrows():
            home_team = match['home_team']
            away_team = match['away_team']
            league = match['league']
            
            # Initialize team stats if not exists, ensure lists for appending
            for team in [home_team, away_team]:
                if team not in self.team_stats:
                    self.team_stats[team] = {
                        'total_matches': 0,
                        'home_matches': 0,
                        'away_matches': 0,
                        'avg_home_odds': [],
                        'avg_away_odds': [],
                        'avg_btts_yes_prob': [],
                        'leagues': set()
                    }
            
            # Update home team stats
            self.team_stats[home_team]['total_matches'] += 1
            self.team_stats[home_team]['home_matches'] += 1
            self.team_stats[home_team]['leagues'].add(league)
            if 'home_win_odds' in match and not pd.isna(match['home_win_odds']):
                self.team_stats[home_team]['avg_home_odds'].append(match['home_win_odds'])
            if 'btts_yes_prob' in match and not pd.isna(match['btts_yes_prob']):
                self.team_stats[home_team]['avg_btts_yes_prob'].append(match['btts_yes_prob'])
            
            # Update away team stats
            self.team_stats[away_team]['total_matches'] += 1
            self.team_stats[away_team]['away_matches'] += 1
            self.team_stats[away_team]['leagues'].add(league)
            if 'away_win_odds' in match and not pd.isna(match['away_win_odds']):
                self.team_stats[away_team]['avg_away_odds'].append(match['away_win_odds'])
            if 'btts_yes_prob' in match and not pd.isna(match['btts_yes_prob']):
                self.team_stats[away_team]['avg_btts_yes_prob'].append(match['btts_yes_prob'])
            
            # Build head-to-head statistics
            h2h_key = f"{home_team}_vs_{away_team}"
            h2h_key_reverse = f"{away_team}_vs_{home_team}" # Consider reverse key for lookup consistency
            
            if h2h_key not in self.h2h_stats:
                self.h2h_stats[h2h_key] = {'matches': 0, 'home_favored': 0, 'total_goals_implied': []}
            if h2h_key_reverse not in self.h2h_stats: # Initialize reverse too if not present
                 self.h2h_stats[h2h_key_reverse] = {'matches': 0, 'home_favored': 0, 'total_goals_implied': []}

            self.h2h_stats[h2h_key]['matches'] += 1
            
            if 'home_win_odds' in match and 'away_win_odds' in match:
                if match['home_win_odds'] < match['away_win_odds']:
                    self.h2h_stats[h2h_key]['home_favored'] += 1
            
            # Add implied total goals from over/under markets
            if 'over_2.5_prob' in match and 'under_2.5_prob' in match:
                implied_goals = 2.5 + (match['over_2.5_prob'] - match['under_2.5_prob']) # Simplified implied goals
                self.h2h_stats[h2h_key]['total_goals_implied'].append(implied_goals)
                # Also update reverse key for consistency in stats
                self.h2h_stats[h2h_key_reverse]['total_goals_implied'].append(implied_goals)

        # Convert lists to averages AFTER all matches are processed
        for team in self.team_stats:
            self.team_stats[team]['avg_home_odds'] = np.mean(self.team_stats[team]['avg_home_odds']) if self.team_stats[team]['avg_home_odds'] else 2.0
            self.team_stats[team]['avg_away_odds'] = np.mean(self.team_stats[team]['avg_away_odds']) if self.team_stats[team]['avg_away_odds'] else 2.0
            self.team_stats[team]['avg_btts_yes_prob'] = np.mean(self.team_stats[team]['avg_btts_yes_prob']) if self.team_stats[team]['avg_btts_yes_prob'] else 0.5
            
            # Calculate team strength indicators
            # Handle potential division by zero
            self.team_stats[team]['home_advantage'] = 1 / self.team_stats[team]['avg_home_odds'] if self.team_stats[team]['avg_home_odds'] != 0 else 0.5
            self.team_stats[team]['away_performance'] = 1 / self.team_stats[team]['avg_away_odds'] if self.team_stats[team]['avg_away_odds'] != 0 else 0.5
            self.team_stats[team]['overall_strength'] = (self.team_stats[team]['home_advantage'] + 
                                                        self.team_stats[team]['away_performance']) / 2
        
        for h2h_key in self.h2h_stats:
            if self.h2h_stats[h2h_key]['total_goals_implied']:
                self.h2h_stats[h2h_key]['h2h_avg_goals'] = np.mean(self.h2h_stats[h2h_key]['total_goals_implied'])
            else:
                self.h2h_stats[h2h_key]['h2h_avg_goals'] = 2.5

    def add_team_features(self, df):
        """Add team-based features to the dataframe"""
        print("Adding team-based features...")
        
        # Iterating directly on df.index for loc assignment
        for idx in df.index:
            home_team = df.loc[idx, 'home_team']
            away_team = df.loc[idx, 'away_team']
            
            # Home team features
            home_stats = self.team_stats.get(home_team, {})
            df.loc[idx, 'home_team_strength'] = home_stats.get('overall_strength', 0.5)
            df.loc[idx, 'home_team_matches'] = home_stats.get('total_matches', 0)
            df.loc[idx, 'home_team_home_advantage'] = home_stats.get('home_advantage', 0.5)
            df.loc[idx, 'home_team_avg_btts'] = home_stats.get('avg_btts_yes_prob', 0.5)
            
            # Away team features
            away_stats = self.team_stats.get(away_team, {})
            df.loc[idx, 'away_team_strength'] = away_stats.get('overall_strength', 0.5)
            df.loc[idx, 'away_team_matches'] = away_stats.get('total_matches', 0)
            df.loc[idx, 'away_team_away_performance'] = away_stats.get('away_performance', 0.5)
            df.loc[idx, 'away_team_avg_btts'] = away_stats.get('avg_btts_yes_prob', 0.5)
            
            # Team comparison features
            df.loc[idx, 'team_strength_diff'] = (home_stats.get('overall_strength', 0.5) - 
                                                      away_stats.get('overall_strength', 0.5))
            df.loc[idx, 'experience_diff'] = (home_stats.get('total_matches', 0) - 
                                                   away_stats.get('total_matches', 0))
            
            # Head-to-head features
            h2h_key = f"{home_team}_vs_{away_team}"
            # Try both permutations for h2h stats
            h2h_stats = self.h2h_stats.get(h2h_key, self.h2h_stats.get(f"{away_team}_vs_{home_team}", {}))
            
            df.loc[idx, 'h2h_matches'] = h2h_stats.get('matches', 0)
            # Use the calculated h2h_avg_goals
            df.loc[idx, 'h2h_avg_goals'] = h2h_stats.get('h2h_avg_goals', 2.5)

            # Home favored percentage needs to be calculated dynamically if teams swap
            home_favored_count = h2h_stats.get('home_favored', 0)
            total_h2h_matches = h2h_stats.get('matches', 0)

            if total_h2h_matches > 0:
                if h2h_key in self.h2h_stats: # current home team was home in h2h
                    df.loc[idx, 'h2h_home_favored_pct'] = home_favored_count / total_h2h_matches
                else: # current home team was away in h2h, so reverse favored count
                    df.loc[idx, 'h2h_home_favored_pct'] = (total_h2h_matches - home_favored_count) / total_h2h_matches
            else:
                df.loc[idx, 'h2h_home_favored_pct'] = 0.5 # Neutral if no H2H
        
        return df
    
    def load_training_data(self):
        """Load and enhance all training JSON files"""
        all_matches = []
        
        if not os.path.exists(self.train_folder):
            print(f"Training folder {self.train_folder} not found.")
            return pd.DataFrame()
        
        json_files = [f for f in os.listdir(self.train_folder) if f.endswith('.json')]
        
        if not json_files:
            print(f"No JSON files found in {self.train_folder}")
            return pd.DataFrame()
        
        print(f"Found {len(json_files)} JSON files to process...")
        
        for filename in json_files:
            filepath = os.path.join(self.train_folder, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    
                matches_in_file = data.get('items', [])
                # print(f"Processing {len(matches_in_file)} matches from {filename}") # Removed for cleaner output
                
                for match in matches_in_file:
                    features = self.extract_features_from_match(match)
                    all_matches.append(features)
                    
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        
        df = pd.DataFrame(all_matches)
        
        if not df.empty:
            # Build team statistics first
            self.build_team_statistics(df.copy()) # Pass a copy to avoid modification during stat building
            
            # Add team-based features
            df = self.add_team_features(df)
            
            print(f"Total matches loaded: {len(df)}")
            print(f"Unique teams: {len(set(df['home_team'].unique()) | set(df['away_team'].unique()))}")
            print(f"Unique leagues: {len(df['league'].unique())}")
        
        return df
    
    def prepare_features(self, df, is_training=False):
        """Prepare features for training or prediction with better handling"""
        if df.empty:
            return df, []
        
        df_copy = df.copy()
        
        # Encode categorical variables with better handling
        categorical_cols = ['home_team', 'away_team', 'league', 'region']
        
        for col in categorical_cols:
            if col in df_copy.columns:
                df_copy[col] = df_copy[col].astype(str) # Ensure string type
                if is_training:
                    # Training phase - fit new encoder
                    unique_values = df_copy[col].unique()
                    self.encoders[col] = LabelEncoder()
                    self.encoders[col].fit(unique_values)
                    df_copy[f'{col}_encoded'] = self.encoders[col].transform(df_copy[col])
                else:
                    # Prediction phase - use existing encoder with unknown handling
                    if col in self.encoders:
                        # Handle unknown categories by mapping to a known placeholder or 0
                        # For prediction, if a category is unknown, we map it to 0 (which is typically the first label)
                        # or to a special 'unknown' label if the encoder was fitted with one.
                        # For simplicity, map unknown to a consistent value (e.g., the most frequent or a placeholder).
                        # Here, we'll try to map to an existing class, or default to 0 if it fails.
                        
                        # Find new, unseen categories
                        new_categories = set(df_copy[col].unique()) - set(self.encoders[col].classes_)
                        
                        if new_categories:
                            print(f"Warning: Unknown categories found in {col}: {new_categories}")
                            # Replace unknown categories with a placeholder (e.g., the first class, or a dummy string)
                            # Ensure the placeholder exists in the encoder's classes_
                            if len(self.encoders[col].classes_) > 0:
                                # Map to the first class label (usually 'Away Team', 'Draw', 'Home Team' if sorted)
                                # Or a specific 'Unknown' if you fitted the encoder with it.
                                # For simplicity, let's map it to a numeric 0, which implies creating an explicit column for it.
                                # A more robust way is to add an 'unknown' category during fitting,
                                # but that requires modifying the fitting process.
                                df_copy.loc[df_copy[col].isin(new_categories), f'{col}_encoded'] = 0 
                                # Then transform known categories
                                df_copy.loc[~df_copy[col].isin(new_categories), f'{col}_encoded'] = self.encoders[col].transform(df_copy.loc[~df_copy[col].isin(new_categories), col])
                            else: # No classes in encoder, assign 0
                                df_copy[f'{col}_encoded'] = 0
                        else: # All categories are known
                            df_copy[f'{col}_encoded'] = self.encoders[col].transform(df_copy[col])

                    else:
                        # If encoder doesn't exist (should not happen if models loaded correctly)
                        print(f"Warning: Encoder for {col} not found during prediction. Assigning 0.")
                        df_copy[f'{col}_encoded'] = 0
            else:
                # If a categorical column is missing from prediction data, add it with default encoded value
                if f'{col}_encoded' not in df_copy.columns:
                    df_copy[f'{col}_encoded'] = 0
        
        # Select feature columns (exclude non-feature columns)
        exclude_cols = categorical_cols + ['match_id', 'start_time', 'actual_winner', 'actual_home_score', 'actual_away_score']
        # self.feature_columns is set during training. For prediction, we use that list.
        # For training, build the list from the current dataframe.
        if is_training:
            self.feature_columns = [col for col in df_copy.columns if col not in exclude_cols]
        
        # Convert to numeric and handle missing values
        for col in self.feature_columns: # Iterate over the *expected* feature columns
            if col in df_copy.columns:
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
                
                if is_training:
                    # Fill missing values with column median and store
                    fill_value = df_copy[col].median()
                    df_copy[col] = df_copy[col].fillna(fill_value)
                    self.fill_values[col] = fill_value
                else:
                    # Use stored fill value from training, or median if not found
                    fill_value = self.fill_values.get(col, df_copy[col].median() if not df_copy[col].empty else 0)
                    df_copy[col] = df_copy[col].fillna(fill_value)
            else:
                # If a feature column expected from training is missing in current df_copy, add it with fill_value
                fill_value = self.fill_values.get(col, 0) # Default to 0 if no fill value stored
                df_copy[col] = fill_value
        
        # Remove columns with too many missing values or zero variance (only during training)
        if is_training:
            valid_cols = []
            for col in self.feature_columns:
                if col in df_copy.columns: # Ensure column exists before checking variance/missing
                    # Check for zero variance
                    if df_copy[col].var() > 1e-10:
                        # Check missing value percentage
                        missing_pct = df_copy[col].isna().sum() / len(df_copy)
                        if missing_pct < 0.8:  # Keep columns with <80% missing
                            valid_cols.append(col)
                    else:
                        print(f"Warning: Feature '{col}' has zero variance and will be removed.")
            self.feature_columns = valid_cols # Update feature_columns based on valid_cols

        # Ensure all columns in self.feature_columns exist in df_copy before scaling
        # and are in the correct order.
        final_df_features = pd.DataFrame(index=df_copy.index)
        for col in self.feature_columns:
            if col in df_copy.columns:
                final_df_features[col] = df_copy[col]
            else:
                # This should ideally be handled by the previous loop adding missing columns
                # but as a safeguard, add with fill value if still missing
                final_df_features[col] = self.fill_values.get(col, 0)

        # Scale features for better performance
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
    
    def create_targets(self, df):
        """Create target variables for different predictions"""
        targets = {}
        
        # Winner prediction (more robust target creation)
        if all(col in df.columns for col in ['home_win_prob', 'away_win_prob', 'draw_prob']):
            probs = df[['away_win_prob', 'draw_prob', 'home_win_prob']].values
            
            # Use argmax but with confidence threshold
            # Note: This uses predicted probabilities from raw data as "truth" for targets.
            # This is a common approach when actual outcomes are not available, assuming
            # the bookmaker probabilities reflect the true underlying probabilities to some extent.
            max_probs = np.max(probs, axis=1)
            confident_mask = max_probs > 0.4  # Only use confident predictions as targets
            
            targets['winner'] = np.argmax(probs, axis=1)
            targets['winner_confident'] = confident_mask
            
            # Over/Under prediction for different goal lines
            for goal_line in [1.5, 2.5, 3.5]:
                over_col = f'over_{goal_line}_prob'
                under_col = f'under_{goal_line}_prob'
                
                if over_col in df.columns and under_col in df.columns:
                    # More conservative threshold
                    over_under_diff = df[over_col] - df[under_col]
                    targets[f'over_under_{goal_line}'] = (over_under_diff > 0.0).astype(int) # Over if over_prob > under_prob
                    targets[f'over_under_{goal_line}_confident'] = (abs(over_under_diff) > 0.1) # Confidence based on difference
            
            # Both teams to score
            if 'btts_yes_prob' in df.columns and 'btts_no_prob' in df.columns:
                btts_diff = df['btts_yes_prob'] - df['btts_no_prob']
                targets['btts'] = (btts_diff > 0.0).astype(int) # BTTS Yes if prob_yes > prob_no
                targets['btts_confident'] = (abs(btts_diff) > 0.1)
        
        return targets
    
    def train_models(self):
        """Train enhanced models with multiple algorithms"""
        print("Loading training data...")
        
        # FIX: Reset team and H2H statistics before (re)building
        self.team_stats = defaultdict(dict)
        self.h2h_stats = defaultdict(dict)
        self.fill_values = {} # Also reset fill values
        
        self.training_data = self.load_training_data()
        
        if self.training_data.empty:
            print("No training data available")
            return
        
        print(f"Loaded {len(self.training_data)} matches for training")
        
        # Prepare features
        X, self.feature_columns = self.prepare_features(self.training_data.copy(), is_training=True)
        
        if X.empty or not self.feature_columns:
            print("No features available for training")
            return
        
        print(f"Using {len(self.feature_columns)} features for training")
        
        # Create targets
        targets = self.create_targets(self.training_data)
        
        if not targets:
            print("No targets available for training")
            return
        
        # Train models for different predictions
        for target_name, y in targets.items():
            if target_name.endswith('_confident'):
                continue  # Skip confidence masks
            
            print(f"\nTraining model for {target_name}...")
            
            # Use confident samples only if available
            confident_mask_name = f'{target_name}_confident'
            if confident_mask_name in targets:
                confident_mask = targets[confident_mask_name]
                X_filtered = X[confident_mask]
                y_filtered = y[confident_mask]
                print(f"Using {sum(confident_mask)}/{len(confident_mask)} confident samples")
            else:
                X_filtered = X
                y_filtered = y
            
            if len(X_filtered) < 20 or len(np.unique(y_filtered)) < 2: # Need at least 2 classes for classification
                print(f"Not enough diverse data for {target_name}, skipping...")
                continue
            
            # Split data
            # Ensure y_filtered is a Series for stratify
            y_filtered_series = pd.Series(y_filtered)
            X_train, X_test, y_train, y_test = train_test_split(
                X_filtered, y_filtered_series, test_size=0.2, random_state=42, stratify=y_filtered_series
            )
            
            # Initialize models to test
            models_to_test = {}
            
            # XGBoost Classifier (usually best for this type of data)
            models_to_test['xgboost'] = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss',
                use_label_encoder=False # Suppress warning
            )
            
            # Random Forest with optimized parameters
            models_to_test['random_forest'] = RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42
            )
            
            # Decision Tree with better parameters
            models_to_test['decision_tree'] = DecisionTreeClassifier(
                max_depth=15,
                min_samples_split=8,
                min_samples_leaf=3,
                max_features='sqrt',
                random_state=42
            )
            
            # Logistic Regression for baseline
            models_to_test['logistic'] = LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced'
            )
            
            # Train and evaluate all models
            best_model = None
            best_score = -1 # Initialize with a low score
            best_name = ""
            
            for model_name, model in models_to_test.items():
                try:
                    # Train model
                    model.fit(X_train, y_train)
                    
                    # Cross-validation
                    cv_scores = cross_val_score(model, X_train, y_train, cv=min(5, len(X_train)//10), scoring='accuracy')
                    cv_mean = cv_scores.mean()
                    
                    # Evaluate on test set
                    test_score = model.score(X_test, y_test)
                    
                    print(f"{model_name}:")
                    print(f"  CV Mean Accuracy: {cv_mean:.3f}, Test Accuracy: {test_score:.3f}")
                    
                    # Select best model based on CV score (more reliable)
                    if cv_mean > best_score:
                        best_score = cv_mean
                        best_model = model
                        best_name = model_name
                
                except Exception as e:
                    print(f"Error training {model_name} for {target_name}: {e}")
            
            if best_model is not None:
                self.models[target_name] = best_model
                print(f"Selected {best_name} for {target_name} (CV Score: {best_score:.3f})")
                
                # Feature importance analysis
                if hasattr(best_model, 'feature_importances_'):
                    importance_df = pd.DataFrame({
                        'feature': self.feature_columns,
                        'importance': best_model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    print(f"Top 5 features for {target_name}:")
                    for _, row in importance_df.head(5).iterrows():
                        print(f"  {row['feature']}: {row['importance']:.3f}")
                elif hasattr(best_model, 'coef_'): # For linear models
                     importance_df = pd.DataFrame({
                        'feature': self.feature_columns,
                        'importance': np.abs(best_model.coef_[0]) # Take abs for magnitude
                    }).sort_values('importance', ascending=False)
                     print(f"Top 5 features for {target_name} (based on coefficients magnitude):")
                     for _, row in importance_df.head(5).iterrows():
                        print(f"  {row['feature']}: {row['importance']:.3f}")

            else:
                print(f"Failed to train any model for {target_name}")
    
    def hyperparameter_tuning(self, X, y, target_name):
        """Perform hyperparameter tuning for XGBoost"""
        print(f"Tuning hyperparameters for {target_name}...")
        
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1, 0.15],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9]
        }
        
        xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)
        
        grid_search = GridSearchCV(
            xgb_model, param_grid, cv=3, scoring='accuracy', n_jobs=-1
        )
        
        grid_search.fit(X, y)
        
        print(f"Best parameters for {target_name}: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.3f}")
        
        return grid_search.best_estimator_
    
    def predict_match(self, home_team, away_team, league):
        """Enhanced prediction for a specific match"""
        if not self.models:
            print("No trained models available. Please train first.")
            return None
        
        # Create a comprehensive template for match data with default values
        now_dt = datetime.now()
        market_features_template = {
            'home_win_odds': 2.0, 'home_win_prob': 0.33, 'home_win_value': 0,
            'home_refund_value': 0, 'home_key_value': 0,
            'away_win_odds': 2.0, 'away_win_prob': 0.33, 'away_win_value': 0,
            'away_refund_value': 0, 'away_key_value': 0,
            'draw_odds': 3.0, 'draw_prob': 0.33, 'draw_win_value': 0,
            'draw_refund_value': 0, 'draw_key_value': 0,
            'btts_yes_odds': 2.0, 'btts_yes_prob': 0.5, 'btts_yes_win_value': 0,
            'btts_no_odds': 2.0, 'btts_no_prob': 0.5, 'btts_no_win_value': 0,
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
            'region': 'Unknown', # Default region
            'match_id': 'PREDICTED_MATCH', # Placeholder
            'start_time': now_dt.isoformat(), # Placeholder
            'hour': now_dt.hour,
            'day_of_week': now_dt.weekday(),
            'month': now_dt.month,
            **market_features_template # Add all market features with defaults
        }
        
        # Try to find similar matches in training data for odds estimation
        similar_matches = self.training_data[
            ((self.training_data['home_team'] == home_team) | 
             (self.training_data['away_team'] == away_team)) &
            (self.training_data['league'] == league)
        ]
        
        market_cols_to_infer = list(market_features_template.keys())

        if not similar_matches.empty:
            for col in market_cols_to_infer:
                if col in similar_matches.columns and not similar_matches[col].isna().all():
                    match_data[col] = similar_matches[col].mean()
        else:
            # Use overall league averages if no direct similar matches
            league_matches = self.training_data[self.training_data['league'] == league]
            if not league_matches.empty:
                for col in market_cols_to_infer:
                    if col in league_matches.columns and not league_matches[col].isna().all():
                        match_data[col] = league_matches[col].mean()
            # Else, the default values initialized in match_data will be used.
        
        # Add team-based features (directly from pre-calculated stats)
        home_stats = self.team_stats.get(home_team, {})
        away_stats = self.team_stats.get(away_team, {})
        
        match_data.update({
            'home_team_strength': home_stats.get('overall_strength', 0.5),
            'home_team_matches': home_stats.get('total_matches', 0),
            'home_team_home_advantage': home_stats.get('home_advantage', 0.5),
            'home_team_avg_btts': home_stats.get('avg_btts_yes_prob', 0.5),
            
            'away_team_strength': away_stats.get('overall_strength', 0.5),
            'away_team_matches': away_stats.get('total_matches', 0),
            'away_team_away_performance': away_stats.get('away_performance', 0.5),
            'away_team_avg_btts': away_stats.get('avg_btts_yes_prob', 0.5),
            
            'team_strength_diff': (home_stats.get('overall_strength', 0.5) - 
                                  away_stats.get('overall_strength', 0.5)),
            'experience_diff': (home_stats.get('total_matches', 0) - 
                               away_stats.get('total_matches', 0))
        })
        
        # Head-to-head features
        h2h_key = f"{home_team}_vs_{away_team}"
        h2h_stats = self.h2h_stats.get(h2h_key, self.h2h_stats.get(f"{away_team}_vs_{home_team}", {})) # Check both permutations
        
        match_data['h2h_matches'] = h2h_stats.get('matches', 0)
        match_data['h2h_avg_goals'] = h2h_stats.get('h2h_avg_goals', 2.5)

        home_favored_count = h2h_stats.get('home_favored', 0)
        total_h2h_matches = h2h_stats.get('matches', 0)

        if total_h2h_matches > 0:
            if h2h_key in self.h2h_stats: # current home team was home in h2h
                match_data['h2h_home_favored_pct'] = home_favored_count / total_h2h_matches
            else: # current home team was away in h2h, so reverse favored count
                match_data['h2h_home_favored_pct'] = (total_h2h_matches - home_favored_count) / total_h2h_matches
        else:
            match_data['h2h_home_favored_pct'] = 0.5 # Neutral if no H2H
        
        # Calculate advanced derived features (using the now more complete match_data)
        derived = self.calculate_derived_features(match_data)
        match_data.update(derived)
        
        # Convert to DataFrame
        match_df = pd.DataFrame([match_data])
        
        # Prepare features for prediction
        # prepare_features will ensure all self.feature_columns are present and scaled
        X, _ = self.prepare_features(match_df, is_training=False)
        
        predictions = {}
        probabilities = {}
        confidence_scores = {}
        
        for target_name, model in self.models.items():
            try:
                pred = model.predict(X)[0]
                
                # Get prediction probabilities
                if hasattr(model, 'predict_proba'):
                    pred_proba = model.predict_proba(X)[0]
                    confidence = max(pred_proba)  # Confidence is the max probability
                else:
                    pred_proba = None
                    confidence = 0.5
                
                predictions[target_name] = pred
                probabilities[target_name] = pred_proba
                confidence_scores[target_name] = confidence
                
            except Exception as e:
                print(f"Error predicting {target_name}: {e}")
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'confidence': confidence_scores,
            'match_info': {
                'home_team': home_team,
                'away_team': away_team,
                'league': league
            },
            'team_stats': {
                'home_strength': home_stats.get('overall_strength', 0.5),
                'away_strength': away_stats.get('overall_strength', 0.5),
                'home_experience': home_stats.get('total_matches', 0),
                'away_experience': away_stats.get('total_matches', 0)
            }
        }
    
    def ensemble_predict(self, home_team, away_team, league):
        """Create ensemble predictions using multiple approaches"""
        if not self.models:
            return None
        
        # Get individual model predictions
        base_prediction = self.predict_match(home_team, away_team, league)
        
        if not base_prediction:
            return None
        
        # Ensemble approach: weight predictions by confidence
        ensemble_results = {}
        
        for target_name in base_prediction['predictions']:
            if target_name in base_prediction['probabilities'] and base_prediction['probabilities'][target_name] is not None:
                probs = base_prediction['probabilities'][target_name]
                confidence = base_prediction['confidence'][target_name]
                
                # Apply confidence weighting (adjust weights as needed)
                if confidence > 0.75: # Very high confidence
                    weight = 1.3
                elif confidence > 0.60: # High confidence
                    weight = 1.1
                elif confidence > 0.5:  # Medium confidence
                    weight = 1.0
                else:  # Low confidence
                    weight = 0.9
                
                # Adjust probabilities based on confidence
                # This is a simplified approach, more complex methods exist (e.g., stacking, blending)
                adjusted_probs = probs * weight
                adjusted_probs = adjusted_probs / (adjusted_probs.sum() + 1e-10)  # Normalize
                
                ensemble_results[target_name] = {
                    'prediction': np.argmax(adjusted_probs),
                    'probabilities': adjusted_probs,
                    'confidence': confidence,
                    'weight': weight
                }
            else:
                ensemble_results[target_name] = {
                    'prediction': base_prediction['predictions'][target_name],
                    'probabilities': None,
                    'confidence': base_prediction['confidence'][target_name],
                    'weight': 1.0 # No adjustment if no probabilities
                }
        
        return {
            'ensemble_predictions': ensemble_results,
            'base_prediction': base_prediction
        }
    
    def validate_model_performance(self):
        """Comprehensive model validation"""
        if self.training_data.empty or not self.models:
            print("No data or models available for validation")
            return
        
        print("\n=== MODEL VALIDATION ===")
        
        # Prepare data
        X, _ = self.prepare_features(self.training_data.copy(), is_training=False)
        targets = self.create_targets(self.training_data)
        
        for target_name, model in self.models.items():
            if target_name not in targets:
                continue
                
            y_true = targets[target_name]
            
            # Use confident samples only if available
            confident_mask_name = f'{target_name}_confident'
            if confident_mask_name in targets:
                confident_mask = targets[confident_mask_name]
                X_val = X[confident_mask]
                y_val = y_true[confident_mask]
            else:
                X_val = X
                y_val = y_true
            
            if len(X_val) == 0 or len(np.unique(y_val)) < 2:
                print(f"\n{target_name.upper()} Performance: Not enough diverse data for validation.")
                continue
            
            try:
                # Predictions
                y_pred = model.predict(X_val)
                accuracy = accuracy_score(y_val, y_pred)
                
                print(f"\n{target_name.upper()} Performance:")
                print(f"Accuracy: {accuracy:.3f}")
                print(f"Samples used: {len(y_val)}")
                
                # Detailed classification report
                print("Classification Report:")
                # Ensure labels are unique in y_val for report
                target_names = [str(label) for label in np.unique(y_val)]
                print(classification_report(y_val, y_pred, target_names=target_names, zero_division=0))
                
                # Confusion matrix for winner prediction
                if target_name == 'winner':
                    cm = confusion_matrix(y_val, y_pred)
                    print("Confusion Matrix (Away Win, Draw, Home Win):")
                    print(cm)
                    
                    # Calculate per-class accuracy
                    class_names = ['Away Win', 'Draw', 'Home Win']
                    for i, class_name in enumerate(class_names):
                        if i < cm.shape[0]: # Check if class exists in matrix
                            class_accuracy = cm[i, i] / max(cm[i].sum(), 1)
                            print(f"{class_name} accuracy: {class_accuracy:.3f}")
                
            except Exception as e:
                print(f"Error validating {target_name}: {e}")
    
    def predict_with_reasoning(self, home_team, away_team, league):
        """Provide predictions with detailed reasoning"""
        result = self.ensemble_predict(home_team, away_team, league)
        
        if not result:
            return None
        
        print(f"\n=== PREDICTION ANALYSIS: {home_team} vs {away_team} ===")
        
        # Team analysis
        base_pred = result['base_prediction']
        home_strength = base_pred['team_stats'].get('home_strength', 0.5)
        away_strength = base_pred['team_stats'].get('away_strength', 0.5)
        
        print(f"\nTeam Strength Analysis:")
        print(f"  {home_team} strength: {home_strength:.3f}")
        print(f"  {away_team} strength: {away_strength:.3f}")
        print(f"  Strength advantage: {home_team if home_strength > away_strength else away_team}")
        
        # Main predictions
        ensemble_preds = result['ensemble_predictions']
        
        if 'winner' in ensemble_preds:
            winner_pred = ensemble_preds['winner']
            winner_map = {0: away_team, 1: 'Draw', 2: home_team}
            predicted_winner = winner_map.get(winner_pred['prediction'], 'Unknown')
            confidence = winner_pred['confidence']
            
            print(f"\nMATCH RESULT PREDICTION:")
            print(f"  Predicted Winner: {predicted_winner}")
            print(f"  Confidence: {confidence:.1%}")
            
            if winner_pred['probabilities'] is not None and len(winner_pred['probabilities']) == 3:
                probs = winner_pred['probabilities']
                print(f"  Probabilities:")
                print(f"    {away_team}: {probs[0]:.1%}")
                print(f"    Draw: {probs[1]:.1%}")
                print(f"    {home_team}: {probs[2]:.1%}")
            
            # Betting recommendation
            if confidence > 0.75:
                print(f"  Recommendation: STRONG BET on {predicted_winner}")
            elif confidence > 0.60:
                print(f"  Recommendation: Consider betting on {predicted_winner}")
            else:
                print(f"  Recommendation: LOW CONFIDENCE - avoid betting")
        
        # Over/Under predictions
        print(f"\nGOALS PREDICTIONS:")
        for target in ensemble_preds:
            if target.startswith('over_under'):
                goal_line = target.split('_')[-1]
                pred_val = ensemble_preds[target]['prediction']
                confidence = ensemble_preds[target]['confidence']
                ou_map = {0: 'Under', 1: 'Over'}
                
                print(f"  {goal_line} goals: {ou_map.get(pred_val, 'Unknown')} (Confidence: {confidence:.1%})")
        
        # BTTS prediction
        if 'btts' in ensemble_preds:
            btts_pred = ensemble_preds['btts']
            btts_result = 'Yes' if btts_pred['prediction'] == 1 else 'No'
            btts_confidence = btts_pred['confidence']
            
            print(f"\nBOTH TEAMS TO SCORE:")
            print(f"  Prediction: {btts_result}")
            print(f"  Confidence: {btts_confidence:.1%}")
        
        return result
    
    def batch_predict(self, matches_list):
        """Predict multiple matches at once"""
        results = []
        
        print(f"Predicting {len(matches_list)} matches...")
        
        for i, (home_team, away_team, league) in enumerate(matches_list):
            print(f"\n--- Predicting match {i+1}/{len(matches_list)}: {home_team} vs {away_team} ({league}) ---")
            result = self.predict_with_reasoning(home_team, away_team, league)
            
            if result:
                results.append({
                    'match': f"{home_team} vs {away_team}",
                    'league': league,
                    'prediction_data': result
                })
        
        return results
    
    def save_models(self, filename='enhanced_betting_models.pkl'):
        """Save trained models and all preprocessing objects"""
        model_data = {
            'models': self.models,
            'encoders': self.encoders,
            'scalers': self.scalers,
            'feature_columns': self.feature_columns,
            'training_data': self.training_data,
            # Convert defaultdicts to dict for pickling
            'team_stats': dict(self.team_stats),
            'h2h_stats': dict(self.h2h_stats),
            'fill_values': self.fill_values
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Enhanced models saved to {filename}")
    
    def load_models(self, filename='enhanced_betting_models.pkl'):
        """Load trained models and all preprocessing objects"""
        try:
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)
            
            self.models = model_data.get('models', {})
            self.encoders = model_data.get('encoders', {})
            self.scalers = model_data.get('scalers', {})
            self.feature_columns = model_data.get('feature_columns', [])
            self.training_data = model_data.get('training_data', pd.DataFrame())
            # Convert dicts back to defaultdicts
            self.team_stats = defaultdict(dict, model_data.get('team_stats', {}))
            self.h2h_stats = defaultdict(dict, model_data.get('h2h_stats', {}))
            self.fill_values = model_data.get('fill_values', {})
            
            print(f"Enhanced models loaded from {filename}")
            print(f"Loaded {len(self.models)} models")
            print(f"Team statistics for {len(self.team_stats)} teams")
            return True
        except FileNotFoundError:
            print(f"Model file {filename} not found")
            return False
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def analyze_league_patterns(self):
        """Analyze patterns specific to each league"""
        if self.training_data.empty:
            print("No training data available for league analysis.")
            return
        
        print("\n=== LEAGUE-SPECIFIC ANALYSIS ===")
        
        for league in self.training_data['league'].unique():
            league_data = self.training_data[self.training_data['league'] == league]
            print(f"\n{league} ({len(league_data)} matches):")
            
            # Average odds analysis
            if 'home_win_odds' in league_data.columns:
                avg_home_odds = league_data['home_win_odds'].mean()
                avg_away_odds = league_data['away_win_odds'].mean()
                avg_draw_odds = league_data['draw_odds'].mean()
                
                print(f"  Average odds - Home: {avg_home_odds:.2f}, Away: {avg_away_odds:.2f}, Draw: {avg_draw_odds:.2f}")
                
                # Home advantage analysis
                home_favored = (league_data['home_win_odds'] < league_data['away_win_odds']).sum()
                home_advantage_pct = home_favored / max(len(league_data), 1) * 100
                print(f"  Home advantage: {home_advantage_pct:.1f}% of matches")
            
            # Goals analysis
            if 'over_2.5_prob' in league_data.columns:
                avg_over_25 = league_data['over_2.5_prob'].mean()
                print(f"  Average Over 2.5 probability: {avg_over_25:.3f}")
            
            # BTTS analysis
            if 'btts_yes_prob' in league_data.columns:
                avg_btts = league_data['btts_yes_prob'].mean()
                print(f"  Average BTTS Yes probability: {avg_btts:.3f}")
    
    def get_prediction_summary(self):
        """Get summary of model performance and recommendations"""
        if not self.models:
            return "No models trained yet."
        if self.training_data.empty:
            return "No training data available to generate summary."
        
        summary = []
        summary.append("=== MODEL PERFORMANCE SUMMARY ===")
        
        # Validate performance
        X, _ = self.prepare_features(self.training_data.copy(), is_training=False)
        targets = self.create_targets(self.training_data)
        
        for target_name, model in self.models.items():
            if target_name in targets:
                try:
                    y_true = targets[target_name]
                    
                    # Use confident samples
                    confident_mask_name = f'{target_name}_confident'
                    if confident_mask_name in targets:
                        mask = targets[confident_mask_name]
                        X_val = X[mask]
                        y_val = y_true[mask]
                    else:
                        X_val = X
                        y_val = y_true
                    
                    if len(X_val) > 0 and len(np.unique(y_val)) > 1:
                        y_pred = model.predict(X_val)
                        accuracy = accuracy_score(y_val, y_pred)
                        
                        if accuracy > 0.7:
                            confidence_level = "HIGH"
                        elif accuracy > 0.6:
                            confidence_level = "MEDIUM"
                        else:
                            confidence_level = "LOW"
                        
                        summary.append(f"{target_name}: {accuracy:.1%} accuracy ({confidence_level} confidence) on {len(X_val)} samples")
                
                except Exception as e:
                    summary.append(f"{target_name}: Error in validation - {e}")
        
        summary.append(f"\nTotal training data: {len(self.training_data)} matches")
        summary.append(f"Teams covered: {len(self.team_stats)}")
        summary.append(f"Leagues covered: {len(self.training_data['league'].unique()) if not self.training_data.empty else 0}")
        
        return "\n".join(summary)

def main():
    """Enhanced main execution function"""
    predictor = EnhancedSportsPredictor()
    
    # Try to load existing models
    if not predictor.load_models():
        print("No existing models found. Training new enhanced models...")
        predictor.train_models()
        if predictor.models:
            predictor.save_models()
    
    # Validate model performance
    predictor.validate_model_performance()
    
    # Analyze league patterns
    predictor.analyze_league_patterns()
    
    # Performance summary
    print(predictor.get_prediction_summary())
    
    # Interactive prediction with enhanced commands
    print("\n=== ENHANCED INTERACTIVE PREDICTION ===")
    print("Available commands:")
    print("1. predict <home_team> <away_team> <league> - Enhanced prediction with reasoning")
    print("2. ensemble <home_team> <away_team> <league> - Ensemble prediction (details in predict)")
    print("3. batch - Predict multiple matches from a list")
    print("4. validate - Run model validation")
    print("5. retrain - Retrain models with current data")
    print("6. summary - Show model performance summary")
    print("7. league_analysis - Show league-specific patterns")
    print("8. team_strength <team_name> - Show team strength analysis")
    print("9. quit")
    
    while True:
        try:
            command = input("\nEnter command: ").strip().split(maxsplit=3) # Limit split to 3 for league name
            
            if not command:
                continue
                
            if command[0] == 'quit':
                break
                
            elif command[0] == 'predict' and len(command) >= 4:
                home_team = command[1]
                away_team = command[2]
                league = command[3] # League is the rest of the string
                predictor.predict_with_reasoning(home_team, away_team, league)
                
            elif command[0] == 'ensemble' and len(command) >= 4:
                home_team = command[1]
                away_team = command[2]
                league = command[3]
                result = predictor.ensemble_predict(home_team, away_team, league)
                if result:
                    print("Ensemble prediction completed. Use 'predict' for detailed reasoning.")
                
            elif command[0] == 'batch':
                print("Enter matches to predict (format: home_team,away_team,league), empty line to finish:")
                matches = []
                while True:
                    match_input = input("Match: ").strip()
                    if not match_input:
                        break
                    parts = [p.strip() for p in match_input.split(',', maxsplit=2)] # Split league name correctly
                    if len(parts) == 3:
                        matches.append((parts[0], parts[1], parts[2]))
                    else:
                        print("Invalid format, use: home_team,away_team,league")
                
                if matches:
                    predictor.batch_predict(matches)
                
            elif command[0] == 'validate':
                predictor.validate_model_performance()
                
            elif command[0] == 'retrain':
                print("Retraining models...")
                predictor.train_models()
                if predictor.models:
                    predictor.save_models()
                
            elif command[0] == 'summary':
                print(predictor.get_prediction_summary())
                
            elif command[0] == 'league_analysis':
                predictor.analyze_league_patterns()
                
            elif command[0] == 'team_strength' and len(command) >= 2:
                team_name = ' '.join(command[1:])
                if team_name in predictor.team_stats:
                    stats = predictor.team_stats[team_name]
                    print(f"\n{team_name} Analysis:")
                    print(f"  Overall strength: {stats.get('overall_strength', 0):.3f}")
                    print(f"  Total matches: {stats.get('total_matches', 0)}")
                    print(f"  Home matches: {stats.get('home_matches', 0)}")
                    print(f"  Away matches: {stats.get('away_matches', 0)}")
                    print(f"  Home advantage: {stats.get('home_advantage', 0):.3f}")
                    print(f"  Away performance: {stats.get('away_performance', 0):.3f}")
                    print(f"  Leagues: {', '.join(stats.get('leagues', set()))}")
                else:
                    print(f"No data found for team: {team_name}")
                    
            else:
                print("Invalid command or insufficient arguments. Type 'help' for commands.")
                
        except KeyboardInterrupt:
            print("\nExiting.")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()