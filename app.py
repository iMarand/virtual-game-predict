import json
import os
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
import pickle
import base64
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SportsPredictor:
    def __init__(self, train_folder='train'):
        self.train_folder = train_folder
        self.models = {}
        self.encoders = {}
        self.scalers = {}
        self.feature_columns = []
        self.training_data = pd.DataFrame()
        
    def decode_probability(self, prob_string):
        """Decode the base64 encoded probability string"""
        try:
            decoded = base64.b64decode(prob_string)
            prob_data = json.loads(decoded.decode('utf-8'))
            
            # Extract win, refund, and key values
            win_value = prob_data.get('win', 0)
            refund_value = prob_data.get('refund', 0)
            key_value = prob_data.get('key', 0)
            
            # Normalize win value to probability (0-1 range)
            # Using absolute values and normalizing
            prob = abs(win_value) / (abs(win_value) + abs(refund_value) + 1e-10)
            prob = max(0.01, min(0.99, prob))  # Clamp between 0.01 and 0.99
            
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
    
    def extract_features_from_match(self, match_data):
        """Extract features from a single match"""
        features = {}
        
        # Basic match info
        features['home_team'] = match_data['participants'][0]['name']
        features['away_team'] = match_data['participants'][1]['name']
        features['league'] = match_data['competition']['name']
        features['region'] = match_data['region']['name']
        
        # Extract odds and probabilities for different markets
        for market in match_data['markets']:
            market_name = market['marketType']['name']
            
            if market_name == '1X2 - FT':
                for row in market['row']:
                    for price in row['prices']:
                        if price['name'] == '1':
                            prob_data = self.decode_probability(price['probability'])
                            features['home_win_odds'] = price['price']
                            features['home_win_prob'] = prob_data['probability']
                            features['home_win_value'] = prob_data['win_value']
                            features['home_refund_value'] = prob_data['refund_value']
                            features['home_key_value'] = prob_data['key_value']
                        elif price['name'] == 'X':
                            prob_data = self.decode_probability(price['probability'])
                            features['draw_odds'] = price['price']
                            features['draw_prob'] = prob_data['probability']
                            features['draw_win_value'] = prob_data['win_value']
                            features['draw_refund_value'] = prob_data['refund_value']
                            features['draw_key_value'] = prob_data['key_value']
                        elif price['name'] == '2':
                            prob_data = self.decode_probability(price['probability'])
                            features['away_win_odds'] = price['price']
                            features['away_win_prob'] = prob_data['probability']
                            features['away_win_value'] = prob_data['win_value']
                            features['away_refund_value'] = prob_data['refund_value']
                            features['away_key_value'] = prob_data['key_value']
            
            elif market_name == 'Total Score Over/Under - FT':
                for row in market['row']:
                    handicap = row.get('handicap', 0)
                    for price in row['prices']:
                        if price['name'] == 'Over':
                            prob_data = self.decode_probability(price['probability'])
                            features[f'over_{handicap/4}_odds'] = price['price']
                            features[f'over_{handicap/4}_prob'] = prob_data['probability']
                            features[f'over_{handicap/4}_win_value'] = prob_data['win_value']
                        elif price['name'] == 'Under':
                            prob_data = self.decode_probability(price['probability'])
                            features[f'under_{handicap/4}_odds'] = price['price']
                            features[f'under_{handicap/4}_prob'] = prob_data['probability']
                            features[f'under_{handicap/4}_win_value'] = prob_data['win_value']
            
            elif market_name == 'Both Teams To Score - FT':
                for row in market['row']:
                    for price in row['prices']:
                        if price['name'] == 'Yes':
                            prob_data = self.decode_probability(price['probability'])
                            features['btts_yes_odds'] = price['price']
                            features['btts_yes_prob'] = prob_data['probability']
                            features['btts_yes_win_value'] = prob_data['win_value']
                        elif price['name'] == 'No':
                            prob_data = self.decode_probability(price['probability'])
                            features['btts_no_odds'] = price['price']
                            features['btts_no_prob'] = prob_data['probability']
                            features['btts_no_win_value'] = prob_data['win_value']
        
        # Calculate derived features
        if 'home_win_odds' in features and 'away_win_odds' in features:
            features['odds_ratio'] = features['home_win_odds'] / features['away_win_odds']
            features['total_prob'] = features['home_win_prob'] + features['away_win_prob'] + features['draw_prob']
            features['home_advantage'] = 1 / features['home_win_odds']
            features['away_advantage'] = 1 / features['away_win_odds']
        
        return features
    
    def load_training_data(self):
        """Load all training JSON files"""
        all_matches = []
        
        if not os.path.exists(self.train_folder):
            print(f"Training folder {self.train_folder} not found.")
            return pd.DataFrame()
        
        json_files = [f for f in os.listdir(self.train_folder) if f.endswith('.json')]
        
        if not json_files:
            print(f"No JSON files found in {self.train_folder}")
            return pd.DataFrame()
        
        for filename in json_files:
            filepath = os.path.join(self.train_folder, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    
                for match in data.get('items', []):
                    features = self.extract_features_from_match(match)
                    features['match_id'] = match['id']
                    features['start_time'] = match['startTime']
                    all_matches.append(features)
                    
                print(f"Loaded {len(data.get('items', []))} matches from {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        
        return pd.DataFrame(all_matches)
    
    def prepare_features(self, df, is_training=False):
        """Prepare features for training or prediction"""
        if df.empty:
            return df, []
        
        df_copy = df.copy()
        
        # Encode categorical variables
        categorical_cols = ['home_team', 'away_team', 'league', 'region']
        
        for col in categorical_cols:
            if col in df_copy.columns:
                if is_training or col not in self.encoders:
                    # Training phase - fit new encoder
                    self.encoders[col] = LabelEncoder()
                    df_copy[f'{col}_encoded'] = self.encoders[col].fit_transform(df_copy[col].astype(str))
                else:
                    # Prediction phase - use existing encoder
                    df_col = df_copy[col].astype(str)
                    
                    # Handle unknown categories
                    known_categories = set(self.encoders[col].classes_)
                    unknown_mask = ~df_col.isin(known_categories)
                    
                    if unknown_mask.any():
                        print(f"Warning: Unknown categories found in {col}: {df_col[unknown_mask].unique()}")
                        # Replace unknown categories with most frequent category
                        most_frequent = self.encoders[col].classes_[0]
                        df_col[unknown_mask] = most_frequent
                    
                    df_copy[f'{col}_encoded'] = self.encoders[col].transform(df_col)
        
        # Select numerical and encoded categorical features
        numerical_cols = [col for col in df_copy.columns 
                         if col not in categorical_cols + ['match_id', 'start_time', 'actual_winner'] 
                         and not col.endswith('_encoded')]
        
        encoded_cols = [col for col in df_copy.columns if col.endswith('_encoded')]
        
        feature_cols = numerical_cols + encoded_cols
        
        # Fill missing values
        for col in feature_cols:
            if col in df_copy.columns:
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
                df_copy[col] = df_copy[col].fillna(df_copy[col].mean())
        
        # Store feature columns if training
        if is_training:
            self.feature_columns = feature_cols
        
        return df_copy[feature_cols], feature_cols
    
    def create_targets(self, df):
        """Create target variables for different predictions"""
        targets = {}
        
        if all(col in df.columns for col in ['home_win_prob', 'away_win_prob', 'draw_prob']):
            # Winner prediction (0: away win, 1: draw, 2: home win)
            probs = df[['away_win_prob', 'draw_prob', 'home_win_prob']].values
            targets['winner'] = np.argmax(probs, axis=1)
            
            # Over/Under prediction for different goal lines
            for goal_line in [1.5, 2.5, 3.5]:
                over_col = f'over_{goal_line}_prob'
                under_col = f'under_{goal_line}_prob'
                
                if over_col in df.columns and under_col in df.columns:
                    targets[f'over_under_{goal_line}'] = (df[over_col] > df[under_col]).astype(int)
            
            # Both teams to score
            if 'btts_yes_prob' in df.columns and 'btts_no_prob' in df.columns:
                targets['btts'] = (df['btts_yes_prob'] > df['btts_no_prob']).astype(int)
        
        return targets
    
    def train_models(self):
        """Train decision tree models"""
        print("Loading training data...")
        self.training_data = self.load_training_data()
        
        if self.training_data.empty:
            print("No training data available")
            return
        
        print(f"Loaded {len(self.training_data)} matches for training")
        
        # Prepare features
        X, self.feature_columns = self.prepare_features(self.training_data.copy(), is_training=True)
        
        if X.empty:
            print("No features available for training")
            return
        
        # Create targets
        targets = self.create_targets(self.training_data)
        
        if not targets:
            print("No targets available for training")
            return
        
        # Train models for different predictions
        for target_name, y in targets.items():
            print(f"\nTraining model for {target_name}...")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train Decision Tree
            dt_model = DecisionTreeClassifier(
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42
            )
            dt_model.fit(X_train, y_train)
            
            # Train Random Forest for comparison
            rf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42
            )
            rf_model.fit(X_train, y_train)
            
            # Evaluate models
            dt_score = dt_model.score(X_test, y_test)
            rf_score = rf_model.score(X_test, y_test)
            
            print(f"Decision Tree accuracy: {dt_score:.3f}")
            print(f"Random Forest accuracy: {rf_score:.3f}")
            
            # Choose better model
            if rf_score > dt_score:
                self.models[target_name] = rf_model
                print(f"Selected Random Forest for {target_name}")
            else:
                self.models[target_name] = dt_model
                print(f"Selected Decision Tree for {target_name}")
            
            # Cross-validation
            cv_scores = cross_val_score(self.models[target_name], X_train, y_train, cv=5)
            print(f"Cross-validation score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    def predict_match(self, home_team, away_team, league):
        """Predict outcome for a specific match"""
        if not self.models:
            print("No trained models available. Please train first.")
            return None
        
        # Find match in training data
        match_data = self.training_data[
            (self.training_data['home_team'] == home_team) & 
            (self.training_data['away_team'] == away_team) & 
            (self.training_data['league'] == league)
        ]
        
        if match_data.empty:
            print(f"No data found for {home_team} vs {away_team} in {league}")
            return None
        
        # Use the first match if multiple found
        match_features = match_data.iloc[0:1].copy()
        
        # Prepare features for prediction (not training)
        X, _ = self.prepare_features(match_features, is_training=False)
        
        # Ensure we have the right columns
        missing_cols = set(self.feature_columns) - set(X.columns)
        if missing_cols:
            print(f"Warning: Missing features: {missing_cols}")
            # Add missing columns with default values
            for col in missing_cols:
                X[col] = 0
        
        # Reorder columns to match training
        X = X[self.feature_columns]
        
        predictions = {}
        probabilities = {}
        
        for target_name, model in self.models.items():
            try:
                pred = model.predict(X)[0]
                pred_proba = model.predict_proba(X)[0] if hasattr(model, 'predict_proba') else None
                
                predictions[target_name] = pred
                probabilities[target_name] = pred_proba
            except Exception as e:
                print(f"Error predicting {target_name}: {e}")
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'match_info': {
                'home_team': home_team,
                'away_team': away_team,
                'league': league
            }
        }
    
    def update_model_with_result(self, home_team, away_team, league, actual_result):
        """Update model with actual match result for continuous learning"""
        if not self.models:
            print("No trained models available")
            return
        
        # Find match in training data
        match_idx = self.training_data[
            (self.training_data['home_team'] == home_team) & 
            (self.training_data['away_team'] == away_team) & 
            (self.training_data['league'] == league)
        ].index
        
        if len(match_idx) == 0:
            print(f"Match not found for updating: {home_team} vs {away_team}")
            return
        
        # Update training data with actual result
        idx = match_idx[0]
        
        # Map result to our encoding
        result_mapping = {
            'home_win': 2,
            'away_win': 0,
            'draw': 1
        }
        
        if actual_result in result_mapping:
            # Add actual result to training data
            self.training_data.loc[idx, 'actual_winner'] = result_mapping[actual_result]
            
            # Retrain models with updated data
            print(f"Updating models with result: {actual_result}")
            self.incremental_train()
    
    def incremental_train(self):
        """Perform incremental training with new results"""
        if 'actual_winner' not in self.training_data.columns:
            return
        
        # Filter data with actual results
        labeled_data = self.training_data.dropna(subset=['actual_winner'])
        
        if len(labeled_data) < 10:  # Need minimum data for retraining
            print("Not enough labeled data for retraining")
            return
        
        # Prepare features and targets
        X, _ = self.prepare_features(labeled_data.copy(), is_training=False)
        X = X[self.feature_columns]  # Ensure column order
        y = labeled_data['actual_winner'].astype(int)
        
        # Retrain winner prediction model
        if 'winner' in self.models:
            print("Retraining winner prediction model...")
            self.models['winner'].fit(X, y)
            
            # Evaluate performance
            score = self.models['winner'].score(X, y)
            print(f"Updated model accuracy: {score:.3f}")
    
    def save_models(self, filename='betting_models.pkl'):
        """Save trained models and encoders"""
        model_data = {
            'models': self.models,
            'encoders': self.encoders,
            'scalers': self.scalers,
            'feature_columns': self.feature_columns,
            'training_data': self.training_data
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Models saved to {filename}")
    
    def load_models(self, filename='betting_models.pkl'):
        """Load trained models and encoders"""
        try:
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)
            
            self.models = model_data['models']
            self.encoders = model_data['encoders']
            self.scalers = model_data['scalers']
            self.feature_columns = model_data['feature_columns']
            self.training_data = model_data['training_data']
            
            print(f"Models loaded from {filename}")
            return True
        except FileNotFoundError:
            print(f"Model file {filename} not found")
            return False
    
    def get_team_stats(self, team_name, league):
        """Get historical statistics for a team"""
        team_matches = self.training_data[
            ((self.training_data['home_team'] == team_name) | 
             (self.training_data['away_team'] == team_name)) &
            (self.training_data['league'] == league)
        ]
        
        if team_matches.empty:
            return None
        
        stats = {
            'total_matches': len(team_matches),
            'avg_home_win_odds': team_matches[team_matches['home_team'] == team_name]['home_win_odds'].mean(),
            'avg_away_win_odds': team_matches[team_matches['away_team'] == team_name]['away_win_odds'].mean(),
        }
        
        if 'btts_yes_prob' in team_matches.columns:
            stats['avg_btts_yes_prob'] = team_matches['btts_yes_prob'].mean()
        
        return stats
    
    def analyze_patterns(self):
        """Analyze patterns in the training data"""
        if self.training_data.empty:
            return
        
        print("\n=== PATTERN ANALYSIS ===")
        
        # League analysis
        print("\nMatches per league:")
        league_counts = self.training_data['league'].value_counts()
        print(league_counts)
        
        # Odds distribution analysis
        numerical_cols = ['home_win_odds', 'away_win_odds', 'draw_odds']
        available_cols = [col for col in numerical_cols if col in self.training_data.columns]
        
        for col in available_cols:
            print(f"\n{col} - Mean: {self.training_data[col].mean():.2f}")
        
        # Feature importance if models are trained
        if 'winner' in self.models and hasattr(self.models['winner'], 'feature_importances_'):
            print("\nFeature Importance (Winner Prediction):")
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.models['winner'].feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(importance_df.head(10))
    
    def list_available_matches(self):
        """List all available matches for prediction"""
        if self.training_data.empty:
            print("No training data available")
            return
        
        print("\n=== AVAILABLE MATCHES FOR PREDICTION ===")
        matches = self.training_data[['home_team', 'away_team', 'league']].drop_duplicates()
        
        for league in matches['league'].unique():
            print(f"\n{league}:")
            league_matches = matches[matches['league'] == league]
            for _, match in league_matches.iterrows():
                print(f"  {match['home_team']} vs {match['away_team']}")

def main():
    """Main execution function"""
    predictor = SportsPredictor()
    
    # Try to load existing models
    if not predictor.load_models():
        print("Training new models...")
        predictor.train_models()
        if predictor.models:  # Only save if training was successful
            predictor.save_models()
    
    # Analyze patterns
    predictor.analyze_patterns()
    
    # List available matches
    predictor.list_available_matches()
    
    # Example prediction if data available
    print("\n=== EXAMPLE PREDICTION ===")
    if not predictor.training_data.empty:
        sample_match = predictor.training_data.iloc[0]
        home_team = sample_match['home_team']
        away_team = sample_match['away_team']
        league = sample_match['league']
        
        result = predictor.predict_match(home_team, away_team, league)
        
        if result:
            print(f"\nPrediction for {home_team} vs {away_team} in {league}:")
            
            for pred_type, pred_value in result['predictions'].items():
                if pred_type == 'winner':
                    winner_map = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}
                    print(f"{pred_type}: {winner_map[pred_value]}")
                elif pred_type.startswith('over_under'):
                    ou_map = {0: 'Under', 1: 'Over'}
                    print(f"{pred_type}: {ou_map[pred_value]}")
                elif pred_type == 'btts':
                    btts_map = {0: 'No', 1: 'Yes'}
                    print(f"Both Teams to Score: {btts_map[pred_value]}")
            
            # Show confidence levels
            if 'winner' in result['probabilities'] and result['probabilities']['winner'] is not None:
                probs = result['probabilities']['winner']
                print(f"Confidence - Away: {probs[0]:.3f}, Draw: {probs[1]:.3f}, Home: {probs[2]:.3f}")
    
    # Interactive prediction loop
    print("\n=== INTERACTIVE PREDICTION ===")
    print("Available commands:")
    print("1. predict <home_team> <away_team> <league>")
    print("2. update <home_team> <away_team> <league> <result>")
    print("   (result: home_win, away_win, or draw)")
    print("3. stats <team_name> <league>")
    print("4. retrain")
    print("5. list - show available matches")
    print("6. quit")
    
    while True:
        try:
            command = input("\nEnter command: ").strip().split()
            
            if not command:
                continue
                
            if command[0] == 'quit':
                break
            elif command[0] == 'list':
                predictor.list_available_matches()
                
            elif command[0] == 'predict' and len(command) >= 4:
                home_team = command[1]
                away_team = command[2]
                league = ' '.join(command[3:])
                
                result = predictor.predict_match(home_team, away_team, league)
                if result:
                    print(f"\nPrediction for {home_team} vs {away_team}:")
                    for pred_type, pred_value in result['predictions'].items():
                        if pred_type == 'winner':
                            winner_map = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}
                            print(f"Match Result: {winner_map[pred_value]}")
                        elif pred_type.startswith('over_under'):
                            goal_line = pred_type.split('_')[-1]
                            ou_map = {0: 'Under', 1: 'Over'}
                            print(f"Goals {goal_line}: {ou_map[pred_value]}")
                        elif pred_type == 'btts':
                            btts_map = {0: 'No', 1: 'Yes'}
                            print(f"Both Teams to Score: {btts_map[pred_value]}")
                    
                    # Show confidence
                    if 'winner' in result['probabilities'] and result['probabilities']['winner'] is not None:
                        probs = result['probabilities']['winner']
                        max_prob = max(probs)
                        print(f"Confidence: {max_prob:.1%}")
            
            elif command[0] == 'update' and len(command) >= 5:
                home_team = command[1]
                away_team = command[2]
                league = command[3]
                result = command[4]  # home_win, away_win, or draw
                
                predictor.update_model_with_result(home_team, away_team, league, result)
                predictor.save_models()
            
            elif command[0] == 'stats' and len(command) >= 3:
                team_name = command[1]
                league = ' '.join(command[2:])
                
                stats = predictor.get_team_stats(team_name, league)
                if stats:
                    print(f"\nStats for {team_name} in {league}:")
                    for key, value in stats.items():
                        if value is not None and not pd.isna(value):
                            print(f"{key}: {value:.3f}" if isinstance(value, float) else f"{key}: {value}")
                else:
                    print(f"No stats found for {team_name} in {league}")
            
            elif command[0] == 'retrain':
                predictor.train_models()
                predictor.save_models()
            
            else:
                print("Invalid command or insufficient arguments")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()