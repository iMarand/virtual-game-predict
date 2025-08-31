import json
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from collections import defaultdict, deque
import pickle
import base64
from typing import Dict, List, Tuple, Optional, Any
import logging

# Advanced ML imports
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    VotingClassifier, StackingClassifier, ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, 
    RandomizedSearchCV, StratifiedKFold, TimeSeriesSplit
)
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, 
    LabelEncoder, PolynomialFeatures
)
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, log_loss, precision_recall_curve
)
from sklearn.feature_selection import (
    SelectKBest, f_classif, RFE, SelectFromModel
)

# Advanced ML libraries
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Deep learning
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Hyperparameter optimization
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# Feature importance and interpretability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

warnings.filterwarnings('ignore')

class AdvancedPatternRecognition:
    """Advanced pattern recognition for sports betting odds and results"""
    
    def __init__(self):
        self.patterns = defaultdict(list)
        self.pattern_accuracy = defaultdict(list)
        self.decision_trees = {}
        
    def analyze_odds_patterns(self, df):
        """Analyze complex patterns in odds movements and outcomes"""
        patterns = {}
        
        # Pattern 1: Odds spread analysis
        if all(col in df.columns for col in ['home_win_odds', 'away_win_odds', 'draw_odds']):
            df['odds_spread'] = df[['home_win_odds', 'away_win_odds', 'draw_odds']].max(axis=1) - \
                               df[['home_win_odds', 'away_win_odds', 'draw_odds']].min(axis=1)
            
            # Pattern: Large spreads often indicate clear favorites
            large_spread_mask = df['odds_spread'] > 3.0
            if 'actual_winner' in df.columns:
                large_spread_accuracy = self.calculate_favorite_accuracy(df[large_spread_mask])
                patterns['large_spread_accuracy'] = large_spread_accuracy
        
        # Pattern 2: Draw odds vs outcome differential
        if all(col in df.columns for col in ['draw_odds', 'home_win_odds', 'away_win_odds']):
            df['draw_vs_winner_diff'] = df['draw_odds'] - df[['home_win_odds', 'away_win_odds']].min(axis=1)
            
            # Your specific pattern: if draw_odds - min(home/away) > 2, analyze outcomes
            high_draw_diff_mask = df['draw_vs_winner_diff'] > 2.0
            if 'actual_winner' in df.columns and high_draw_diff_mask.sum() > 0:
                high_diff_outcomes = df[high_draw_diff_mask]['actual_winner'].value_counts()
                patterns['high_draw_diff_pattern'] = {
                    'total_matches': high_draw_diff_mask.sum(),
                    'outcomes': high_diff_outcomes.to_dict(),
                    'non_draw_rate': (high_diff_outcomes.get(0, 0) + high_diff_outcomes.get(2, 0)) / high_draw_diff_mask.sum()
                }
        
        # Pattern 3: Odds efficiency patterns
        if all(col in df.columns for col in ['home_win_odds', 'away_win_odds', 'draw_odds']):
            # Calculate implied probabilities
            df['home_implied'] = 1 / df['home_win_odds']
            df['away_implied'] = 1 / df['away_win_odds']
            df['draw_implied'] = 1 / df['draw_odds']
            df['total_implied'] = df['home_implied'] + df['away_implied'] + df['draw_implied']
            df['bookmaker_margin'] = df['total_implied'] - 1
            
            # Pattern: High margin markets vs low margin markets
            high_margin_mask = df['bookmaker_margin'] > 0.1
            if 'actual_winner' in df.columns:
                high_margin_accuracy = self.calculate_favorite_accuracy(df[high_margin_mask])
                low_margin_accuracy = self.calculate_favorite_accuracy(df[~high_margin_mask])
                patterns['margin_efficiency'] = {
                    'high_margin_accuracy': high_margin_accuracy,
                    'low_margin_accuracy': low_margin_accuracy
                }
        
        # Pattern 4: Time-based patterns
        if 'start_time' in df.columns:
            df['start_datetime'] = pd.to_datetime(df['start_time'])
            df['hour'] = df['start_datetime'].dt.hour
            df['day_of_week'] = df['start_datetime'].dt.dayofweek
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            
            # Weekend vs weekday patterns
            if 'actual_winner' in df.columns:
                weekend_accuracy = self.calculate_favorite_accuracy(df[df['is_weekend'] == 1])
                weekday_accuracy = self.calculate_favorite_accuracy(df[df['is_weekend'] == 0])
                patterns['time_patterns'] = {
                    'weekend_accuracy': weekend_accuracy,
                    'weekday_accuracy': weekday_accuracy
                }
        
        return patterns
    
    def calculate_favorite_accuracy(self, df):
        """Calculate how often the favorite (lowest odds) wins"""
        if df.empty or 'actual_winner' not in df.columns:
            return 0.0
        
        correct = 0
        total = 0
        
        for _, match in df.iterrows():
            if all(col in match for col in ['home_win_odds', 'away_win_odds', 'draw_odds']):
                odds = [match['away_win_odds'], match['draw_odds'], match['home_win_odds']]
                favorite = np.argmin(odds)
                actual = match['actual_winner']
                
                if actual in [0, 1, 2]:
                    total += 1
                    if favorite == actual:
                        correct += 1
        
        return correct / total if total > 0 else 0.0
    
    def build_decision_trees(self, df):
        """Build decision trees for specific patterns"""
        if 'actual_winner' in df.columns:
            # Decision tree for your specific pattern
            if all(col in df.columns for col in ['draw_odds', 'home_win_odds', 'away_win_odds']):
                df['draw_vs_winner_diff'] = df['draw_odds'] - df[['home_win_odds', 'away_win_odds']].min(axis=1)
                
                # Create binary target: non-draw outcome when diff > 2
                high_diff_mask = df['draw_vs_winner_diff'] > 2.0
                if high_diff_mask.sum() > 10:  # Enough samples
                    X = df[high_diff_mask][['draw_vs_winner_diff', 'odds_spread', 'bookmaker_margin']].fillna(0)
                    y = (df[high_diff_mask]['actual_winner'] != 1).astype(int)  # 1 if not draw
                    
                    if len(np.unique(y)) > 1:
                        from sklearn.tree import DecisionTreeClassifier
                        tree = DecisionTreeClassifier(max_depth=5, min_samples_split=5, random_state=42)
                        tree.fit(X, y)
                        self.decision_trees['high_draw_diff'] = tree
                        
                        # Calculate pattern accuracy
                        accuracy = tree.score(X, y)
                        self.pattern_accuracy['high_draw_diff'].append(accuracy)

class AdvancedFeatureEngineering:
    """Advanced feature engineering with domain expertise"""
    
    def __init__(self):
        self.feature_generators = []
        self.interaction_features = []
        
    def create_advanced_features(self, df):
        """Create sophisticated features based on domain knowledge"""
        df_enhanced = df.copy()
        
        # 1. Advanced odds features
        if all(col in df.columns for col in ['home_win_odds', 'away_win_odds', 'draw_odds']):
            # Odds ratios and relationships
            df_enhanced['home_away_ratio'] = df['home_win_odds'] / df['away_win_odds']
            df_enhanced['draw_favorite_ratio'] = df['draw_odds'] / df[['home_win_odds', 'away_win_odds']].min(axis=1)
            
            # Odds momentum (if we have historical odds)
            df_enhanced['odds_variance'] = df[['home_win_odds', 'away_win_odds', 'draw_odds']].var(axis=1)
            df_enhanced['odds_entropy'] = self.calculate_odds_entropy(df)
            
            # Market efficiency indicators
            df_enhanced['market_efficiency'] = self.calculate_market_efficiency(df)
            
        # 2. Advanced probability features
        if all(col in df.columns for col in ['home_win_prob', 'away_win_prob', 'draw_prob']):
            # Probability distributions
            df_enhanced['prob_gini'] = self.calculate_gini_coefficient(df[['home_win_prob', 'away_win_prob', 'draw_prob']])
            df_enhanced['prob_concentration'] = self.calculate_concentration_index(df[['home_win_prob', 'away_win_prob', 'draw_prob']])
            
        # 3. Advanced temporal features
        if 'start_time' in df.columns:
            df_enhanced = self.add_temporal_features(df_enhanced)
            
        # 4. Advanced team performance features
        df_enhanced = self.add_performance_momentum_features(df_enhanced)
        
        # 5. Market psychology features
        df_enhanced = self.add_market_psychology_features(df_enhanced)
        
        return df_enhanced
    
    def calculate_odds_entropy(self, df):
        """Calculate entropy of odds distribution"""
        entropy_values = []
        for _, row in df.iterrows():
            if all(col in row for col in ['home_win_odds', 'away_win_odds', 'draw_odds']):
                # Convert odds to probabilities
                probs = [1/row['home_win_odds'], 1/row['away_win_odds'], 1/row['draw_odds']]
                total_prob = sum(probs)
                normalized_probs = [p/total_prob for p in probs]
                
                # Calculate entropy
                entropy = -sum(p * np.log(p + 1e-10) for p in normalized_probs)
                entropy_values.append(entropy)
            else:
                entropy_values.append(0)
        return entropy_values
    
    def calculate_market_efficiency(self, df):
        """Calculate market efficiency score"""
        efficiency_scores = []
        for _, row in df.iterrows():
            if all(col in row for col in ['home_win_odds', 'away_win_odds', 'draw_odds']):
                odds = [row['home_win_odds'], row['away_win_odds'], row['draw_odds']]
                
                # Market efficiency based on odds balance and margin
                implied_probs = [1/odd for odd in odds]
                total_implied = sum(implied_probs)
                margin = total_implied - 1
                
                # Lower margin = higher efficiency
                efficiency = 1 / (1 + margin * 10)  # Scale to 0-1
                efficiency_scores.append(efficiency)
            else:
                efficiency_scores.append(0.5)
        return efficiency_scores
    
    def calculate_gini_coefficient(self, prob_df):
        """Calculate Gini coefficient for probability distribution"""
        gini_values = []
        for _, row in prob_df.iterrows():
            probs = sorted([row['home_win_prob'], row['away_win_prob'], row['draw_prob']])
            n = len(probs)
            cumsum = np.cumsum(probs)
            gini = (n + 1 - 2 * sum((n + 1 - i) * prob for i, prob in enumerate(probs))) / (n * sum(probs))
            gini_values.append(gini)
        return gini_values
    
    def calculate_concentration_index(self, prob_df):
        """Calculate concentration index (Herfindahl-Hirschman Index)"""
        concentration_values = []
        for _, row in prob_df.iterrows():
            probs = [row['home_win_prob'], row['away_win_prob'], row['draw_prob']]
            concentration = sum(p**2 for p in probs)
            concentration_values.append(concentration)
        return concentration_values
    
    def add_temporal_features(self, df):
        """Add advanced temporal features"""
        if 'start_time' in df.columns:
            df['start_datetime'] = pd.to_datetime(df['start_time'])
            
            # Cyclical encoding for time features
            df['hour_sin'] = np.sin(2 * np.pi * df['start_datetime'].dt.hour / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['start_datetime'].dt.hour / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['start_datetime'].dt.dayofweek / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['start_datetime'].dt.dayofweek / 7)
            df['month_sin'] = np.sin(2 * np.pi * df['start_datetime'].dt.month / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['start_datetime'].dt.month / 12)
            
            # Season features
            df['is_holiday_season'] = ((df['start_datetime'].dt.month == 12) | 
                                      (df['start_datetime'].dt.month == 1)).astype(int)
            df['is_summer'] = ((df['start_datetime'].dt.month >= 6) & 
                              (df['start_datetime'].dt.month <= 8)).astype(int)
            
        return df
    
    def add_performance_momentum_features(self, df):
        """Add momentum and streak features"""
        # These would be calculated from team statistics
        momentum_features = [
            'home_win_streak', 'away_win_streak', 'home_goal_streak',
            'away_goal_streak', 'home_clean_sheet_streak', 'away_clean_sheet_streak'
        ]
        
        for feature in momentum_features:
            if feature not in df.columns:
                df[feature] = 0  # Default value
                
        return df
    
    def add_market_psychology_features(self, df):
        """Add features based on market psychology"""
        if all(col in df.columns for col in ['home_win_odds', 'away_win_odds', 'draw_odds']):
            # Public betting bias indicators
            df['home_bias'] = np.where(df['home_win_odds'] < 1.8, 1, 0)  # Heavy home favorite
            df['away_bias'] = np.where(df['away_win_odds'] < 1.8, 1, 0)  # Heavy away favorite
            df['draw_value'] = np.where(df['draw_odds'] > 4.0, 1, 0)  # High draw odds
            
            # Contrarian indicators
            df['contrarian_signal'] = np.where(
                (df['home_win_odds'] > 3.0) & (df['away_win_odds'] > 3.0), 1, 0
            )  # Both teams undervalued
            
        return df

class AdvancedEnsemblePredictor:
    """Advanced ensemble predictor with multiple algorithms"""
    
    def __init__(self):
        self.base_models = {}
        self.meta_models = {}
        self.feature_selectors = {}
        self.hyperparameter_optimizers = {}
        
    def create_base_models(self):
        """Create diverse base models for ensemble"""
        models = {}
        
        # 1. Gradient Boosting variants
        models['xgboost'] = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        )
        
        models['lightgbm'] = lgb.LGBMClassifier(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            verbose=-1
        )
        
        models['catboost'] = cb.CatBoostClassifier(
            iterations=500,
            depth=8,
            learning_rate=0.05,
            random_seed=42,
            verbose=False
        )
        
        # 2. Tree-based models
        models['random_forest'] = RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=3,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            class_weight='balanced'
        )
        
        models['extra_trees'] = ExtraTreesClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=3,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            class_weight='balanced'
        )
        
        # 3. Linear models
        models['logistic'] = LogisticRegression(
            max_iter=2000,
            random_state=42,
            class_weight='balanced',
            C=0.1,
            solver='liblinear'
        )
        
        models['ridge'] = RidgeClassifier(
            alpha=1.0,
            random_state=42,
            class_weight='balanced'
        )
        
        # 4. Support Vector Machine
        models['svm'] = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            random_state=42,
            class_weight='balanced'
        )
        
        # 5. Neural Network
        models['mlp'] = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=1000,
            random_state=42
        )
        
        return models
    
    def optimize_hyperparameters(self, X, y, model_name, model):
        """Optimize hyperparameters using advanced techniques"""
        if not OPTUNA_AVAILABLE:
            return model
        
        def objective(trial):
            if model_name == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
                }
                model_opt = xgb.XGBClassifier(**params, random_state=42, eval_metric='logloss', use_label_encoder=False)
                
            elif model_name == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 5, 25),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                }
                model_opt = RandomForestClassifier(**params, random_state=42, class_weight='balanced')
                
            else:
                return 0.5  # Skip optimization for other models
            
            # Cross-validation
            cv_scores = cross_val_score(model_opt, X, y, cv=3, scoring='accuracy')
            return cv_scores.mean()
        
        try:
            study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
            study.optimize(objective, n_trials=50, timeout=300)  # 5 minutes max
            
            # Apply best parameters
            if model_name == 'xgboost':
                return xgb.XGBClassifier(**study.best_params, random_state=42, eval_metric='logloss', use_label_encoder=False)
            elif model_name == 'random_forest':
                return RandomForestClassifier(**study.best_params, random_state=42, class_weight='balanced')
                
        except Exception as e:
            print(f"Hyperparameter optimization failed for {model_name}: {e}")
            
        return model
    
    def create_stacked_ensemble(self, base_models, X, y):
        """Create advanced stacked ensemble"""
        # Level 1: Base models
        level1_models = [(name, model) for name, model in base_models.items()]
        
        # Level 2: Meta-learner
        meta_learner = LogisticRegression(random_state=42, class_weight='balanced')
        
        # Create stacking classifier
        stacked_model = StackingClassifier(
            estimators=level1_models,
            final_estimator=meta_learner,
            cv=5,
            stack_method='predict_proba',
            n_jobs=-1
        )
        
        return stacked_model
    
    def create_voting_ensemble(self, base_models):
        """Create voting ensemble with optimized weights"""
        # Soft voting for probability-based predictions
        voting_model = VotingClassifier(
            estimators=[(name, model) for name, model in base_models.items()],
            voting='soft',
            n_jobs=-1
        )
        
        return voting_model

class AdvancedSportsPredictor:
    """Advanced sports predictor with state-of-the-art ML techniques"""
    
    def __init__(self, train_folder='train', results_folder='previousMatch'):
        self.train_folder = train_folder
        self.results_folder = results_folder
        
        # Core components
        self.models = {}
        self.ensemble_models = {}
        self.encoders = {}
        self.scalers = {}
        self.feature_selectors = {}
        
        # Data storage
        self.training_data = pd.DataFrame()
        self.results_data = pd.DataFrame()
        self.matched_data = pd.DataFrame()
        self.upcoming_data = pd.DataFrame()
        
        # Advanced components
        self.pattern_recognizer = AdvancedPatternRecognition()
        self.feature_engineer = AdvancedFeatureEngineering()
        self.ensemble_predictor = AdvancedEnsemblePredictor()
        
        # Statistics and tracking
        self.team_stats = defaultdict(dict)
        self.h2h_stats = defaultdict(dict)
        self.league_stats = defaultdict(dict)
        self.prediction_accuracy = defaultdict(list)
        self.feature_importance_history = defaultdict(list)
        
        # Configuration
        self.feature_columns = []
        self.fill_values = {}
        self.confidence_threshold = 0.65
        self.ensemble_weights = {}
        
        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def decode_probability(self, prob_string):
        """Enhanced probability decoder with error handling"""
        try:
            decoded = base64.b64decode(prob_string)
            prob_data = json.loads(decoded.decode('utf-8'))
            
            win_value = prob_data.get('win', 0)
            refund_value = prob_data.get('refund', 0)
            key_value = prob_data.get('key', 0)
            
            # Enhanced probability calculation
            total = abs(win_value) + abs(refund_value) + abs(key_value) + 1e-10
            prob = abs(win_value) / total
            
            # Apply sigmoid transformation for better distribution
            prob = 1 / (1 + np.exp(-5 * (prob - 0.5)))
            prob = max(0.001, min(0.999, prob))
            
            return {
                'probability': prob,
                'win_value': win_value,
                'refund_value': refund_value,
                'key_value': key_value,
                'confidence': abs(win_value) / (abs(win_value) + abs(refund_value) + 1e-10)
            }
        except Exception as e:
            return {
                'probability': 0.5,
                'win_value': 0,
                'refund_value': 0,
                'key_value': 0,
                'confidence': 0.0
            }
    
    def extract_advanced_features(self, match_data):
        """Extract comprehensive features with advanced engineering"""
        features = {}
        
        # Basic match information
        features['home_team'] = match_data['participants'][0]['name']
        features['away_team'] = match_data['participants'][1]['name']
        features['league'] = match_data['competition']['name']
        features['region'] = match_data['region']['name']
        features['match_id'] = match_data['id']
        features['start_time'] = match_data['startTime']
        
        # Advanced temporal features
        try:
            start_dt = datetime.fromisoformat(match_data['startTime'].replace('Z', '+00:00'))
            features.update(self.extract_temporal_features(start_dt))
        except:
            features.update(self.get_default_temporal_features())
        
        # Initialize comprehensive market features
        market_features = self.initialize_market_features()
        
        # Extract market data with enhanced processing
        for market in match_data.get('markets', []):
            market_name = market['marketType']['name']
            
            if market_name == '1X2 - FT':
                self.process_1x2_market(market, market_features)
            elif market_name == 'Total Score Over/Under - FT':
                self.process_ou_market(market, market_features)
            elif market_name == 'Both Teams To Score - FT':
                self.process_btts_market(market, market_features)
        
        features.update(market_features)
        
        # Advanced derived features
        derived_features = self.calculate_advanced_derived_features(features)
        features.update(derived_features)
        
        return features
    
    def extract_temporal_features(self, start_dt):
        """Extract comprehensive temporal features"""
        return {
            'hour': start_dt.hour,
            'day_of_week': start_dt.weekday(),
            'month': start_dt.month,
            'is_weekend': 1 if start_dt.weekday() >= 5 else 0,
            'is_evening': 1 if start_dt.hour >= 18 else 0,
            'is_prime_time': 1 if 19 <= start_dt.hour <= 21 else 0,
            'is_afternoon': 1 if 12 <= start_dt.hour <= 17 else 0,
            'quarter_of_year': (start_dt.month - 1) // 3 + 1,
            'week_of_year': start_dt.isocalendar()[1],
            'is_month_start': 1 if start_dt.day <= 7 else 0,
            'is_month_end': 1 if start_dt.day >= 24 else 0,
        }
    
    def get_default_temporal_features(self):
        """Get default temporal features"""
        return {
            'hour': 15, 'day_of_week': 2, 'month': 6, 'is_weekend': 0,
            'is_evening': 0, 'is_prime_time': 0, 'is_afternoon': 1,
            'quarter_of_year': 2, 'week_of_year': 26, 'is_month_start': 0,
            'is_month_end': 0
        }
    
    def initialize_market_features(self):
        """Initialize comprehensive market features"""
        features = {}
        
        # 1X2 market
        for outcome in ['home_win', 'away_win', 'draw']:
            features.update({
                f'{outcome}_odds': 2.0,
                f'{outcome}_prob': 0.33,
                f'{outcome}_confidence': 0.5,
                f'{outcome}_value': 0,
                f'{outcome}_refund_value': 0,
                f'{outcome}_key_value': 0
            })
        
        # Over/Under markets
        for line in [1.5, 2.5, 3.5]:
            for direction in ['over', 'under']:
                features.update({
                    f'{direction}_{line}_odds': 2.0,
                    f'{direction}_{line}_prob': 0.5,
                    f'{direction}_{line}_confidence': 0.5,
                    f'{direction}_{line}_value': 0
                })
        
        # BTTS market
        for outcome in ['btts_yes', 'btts_no']:
            features.update({
                f'{outcome}_odds': 2.0,
                f'{outcome}_prob': 0.5,
                f'{outcome}_confidence': 0.5,
                f'{outcome}_value': 0
            })
        
        return features
    
    def process_1x2_market(self, market, features):
        """Process 1X2 market with enhanced feature extraction"""
        for row in market['row']:
            for price in row['prices']:
                prob_data = self.decode_probability(price['probability'])
                
                if price['name'] == '1':  # Home win
                    features.update({
                        'home_win_odds': float(price['price']),
                        'home_win_prob': prob_data['probability'],
                        'home_win_confidence': prob_data['confidence'],
                        'home_win_value': prob_data['win_value'],
                        'home_refund_value': prob_data['refund_value'],
                        'home_key_value': prob_data['key_value']
                    })
                elif price['name'] == 'X':  # Draw
                    features.update({
                        'draw_odds': float(price['price']),
                        'draw_prob': prob_data['probability'],
                        'draw_confidence': prob_data['confidence'],
                        'draw_value': prob_data['win_value'],
                        'draw_refund_value': prob_data['refund_value'],
                        'draw_key_value': prob_data['key_value']
                    })
                elif price['name'] == '2':  # Away win
                    features.update({
                        'away_win_odds': float(price['price']),
                        'away_win_prob': prob_data['probability'],
                        'away_win_confidence': prob_data['confidence'],
                        'away_win_value': prob_data['win_value'],
                        'away_refund_value': prob_data['refund_value'],
                        'away_key_value': prob_data['key_value']
                    })
    
    def process_ou_market(self, market, features):
        """Process Over/Under market with enhanced features"""
        for row in market['row']:
            handicap = row.get('handicap', 0)
            goal_line = handicap / 4
            
            if goal_line in [1.5, 2.5, 3.5]:
                for price in row['prices']:
                    prob_data = self.decode_probability(price['probability'])
                    
                    if price['name'] == 'Over':
                        features.update({
                            f'over_{goal_line}_odds': float(price['price']),
                            f'over_{goal_line}_prob': prob_data['probability'],
                            f'over_{goal_line}_confidence': prob_data['confidence'],
                            f'over_{goal_line}_value': prob_data['win_value']
                        })
                    elif price['name'] == 'Under':
                        features.update({
                            f'under_{goal_line}_odds': float(price['price']),
                            f'under_{goal_line}_prob': prob_data['probability'],
                            f'under_{goal_line}_confidence': prob_data['confidence'],
                            f'under_{goal_line}_value': prob_data['win_value']
                        })
    
    def process_btts_market(self, market, features):
        """Process BTTS market with enhanced features"""
        for row in market['row']:
            for price in row['prices']:
                prob_data = self.decode_probability(price['probability'])
                
                if price['name'] == 'Yes':
                    features.update({
                        'btts_yes_odds': float(price['price']),
                        'btts_yes_prob': prob_data['probability'],
                        'btts_yes_confidence': prob_data['confidence'],
                        'btts_yes_value': prob_data['win_value']
                    })
                elif price['name'] == 'No':
                    features.update({
                        'btts_no_odds': float(price['price']),
                        'btts_no_prob': prob_data['probability'],
                        'btts_no_confidence': prob_data['confidence'],
                        'btts_no_value': prob_data['win_value']
                    })
    
    def calculate_advanced_derived_features(self, features):
        """Calculate sophisticated derived features"""
        derived = {}
        
        # 1. Advanced odds analysis
        if all(k in features for k in ['home_win_odds', 'away_win_odds', 'draw_odds']):
            odds_array = np.array([features['home_win_odds'], features['away_win_odds'], features['draw_odds']])
            
            # Statistical measures
            derived['odds_mean'] = np.mean(odds_array)
            derived['odds_std'] = np.std(odds_array)
            derived['odds_cv'] = derived['odds_std'] / derived['odds_mean'] if derived['odds_mean'] > 0 else 0
            derived['odds_range'] = np.max(odds_array) - np.min(odds_array)
            derived['odds_iqr'] = np.percentile(odds_array, 75) - np.percentile(odds_array, 25)
            
            # Market structure analysis
            derived['favorite_odds'] = np.min(odds_array)
            derived['underdog_odds'] = np.max(odds_array)
            derived['favorite_strength'] = derived['underdog_odds'] / derived['favorite_odds']
            
            # Your specific pattern: draw vs winner difference
            min_winner_odds = min(features['home_win_odds'], features['away_win_odds'])
            derived['draw_winner_diff'] = features['draw_odds'] - min_winner_odds
            derived['draw_winner_ratio'] = features['draw_odds'] / min_winner_odds
            
            # Pattern indicator for your rule
            derived['high_draw_diff_pattern'] = 1 if derived['draw_winner_diff'] > 2.0 else 0
            
            # Market efficiency
            implied_probs = [1/odd for odd in odds_array]
            total_implied = sum(implied_probs)
            derived['bookmaker_margin'] = total_implied - 1
            derived['market_efficiency'] = 1 / (1 + derived['bookmaker_margin'] * 10)
            
            # Probability normalization
            if total_implied > 0:
                normalized_probs = [p/total_implied for p in implied_probs]
                derived['home_norm_prob'] = normalized_probs[0]
                derived['away_norm_prob'] = normalized_probs[1]
                derived['draw_norm_prob'] = normalized_probs[2]
                
                # Entropy and concentration
                derived['prob_entropy'] = -sum(p * np.log(p + 1e-10) for p in normalized_probs)
                derived['prob_concentration'] = sum(p**2 for p in normalized_probs)
        
        # 2. Advanced probability features
        if all(k in features for k in ['home_win_prob', 'away_win_prob', 'draw_prob']):
            prob_array = np.array([features['home_win_prob'], features['away_win_prob'], features['draw_prob']])
            
            # Statistical measures
            derived['prob_variance'] = np.var(prob_array)
            derived['prob_skewness'] = self.calculate_skewness(prob_array)
            derived['prob_kurtosis'] = self.calculate_kurtosis(prob_array)
            
            # Dominance measures
            derived['prob_dominance'] = np.max(prob_array) - np.mean(prob_array)
            derived['prob_balance'] = 1 - derived['prob_dominance']  # Higher = more balanced
        
        # 3. Advanced Over/Under features
        for line in [1.5, 2.5, 3.5]:
            over_key = f'over_{line}_prob'
            under_key = f'under_{line}_prob'
            
            if over_key in features and under_key in features:
                # Market sentiment
                derived[f'ou_{line}_sentiment'] = features[over_key] - features[under_key]
                derived[f'ou_{line}_confidence'] = abs(derived[f'ou_{line}_sentiment'])
                
                # Market efficiency for O/U
                ou_total_prob = features[over_key] + features[under_key]
                derived[f'ou_{line}_efficiency'] = abs(1 - ou_total_prob)
                
                # Volatility indicator
                derived[f'ou_{line}_volatility'] = abs(features[over_key] - 0.5) + abs(features[under_key] - 0.5)
        
        # 4. Advanced BTTS features
        if 'btts_yes_prob' in features and 'btts_no_prob' in features:
            derived['btts_sentiment'] = features['btts_yes_prob'] - features['btts_no_prob']
            derived['btts_market_confidence'] = abs(derived['btts_sentiment'])
            derived['btts_efficiency'] = abs(1 - (features['btts_yes_prob'] + features['btts_no_prob']))
        
        # 5. Cross-market consistency features
        if all(k in features for k in ['home_win_prob', 'away_win_prob', 'over_2.5_prob', 'btts_yes_prob']):
            # Consistency between winner and goals markets
            expected_goals_from_winner = self.estimate_goals_from_winner_probs(
                features['home_win_prob'], features['away_win_prob'], features['draw_prob']
            )
            
            derived['goals_winner_consistency'] = abs(expected_goals_from_winner - (features['over_2.5_prob'] * 3.5))
            
            # BTTS consistency with winner probabilities
            expected_btts_from_winner = self.estimate_btts_from_winner_probs(
                features['home_win_prob'], features['away_win_prob'], features['draw_prob']
            )
            
            derived['btts_winner_consistency'] = abs(expected_btts_from_winner - features['btts_yes_prob'])
        
        return derived
    
    def calculate_skewness(self, array):
        """Calculate skewness of array"""
        mean = np.mean(array)
        std = np.std(array)
        if std == 0:
            return 0
        return np.mean(((array - mean) / std) ** 3)
    
    def calculate_kurtosis(self, array):
        """Calculate kurtosis of array"""
        mean = np.mean(array)
        std = np.std(array)
        if std == 0:
            return 0
        return np.mean(((array - mean) / std) ** 4) - 3
    
    def estimate_goals_from_winner_probs(self, home_prob, away_prob, draw_prob):
        """Estimate expected goals from winner probabilities"""
        # Empirical relationship: higher win probabilities suggest more goals
        attacking_strength = (home_prob + away_prob) / (home_prob + away_prob + draw_prob)
        return 1.5 + attacking_strength * 2.0  # Scale to realistic goal range
    
    def estimate_btts_from_winner_probs(self, home_prob, away_prob, draw_prob):
        """Estimate BTTS probability from winner probabilities"""
        # Higher individual win probabilities suggest both teams can score
        individual_strength = min(home_prob, away_prob)
        return min(0.9, max(0.1, individual_strength * 2.0))
    
    def build_advanced_team_statistics(self, df):
        """Build comprehensive team statistics with advanced metrics"""
        print("Building advanced team statistics...")
        
        # Reset statistics
        self.team_stats = defaultdict(lambda: {
            # Basic stats
            'total_matches': 0, 'wins': 0, 'draws': 0, 'losses': 0,
            'home_matches': 0, 'home_wins': 0, 'home_draws': 0, 'home_losses': 0,
            'away_matches': 0, 'away_wins': 0, 'away_draws': 0, 'away_losses': 0,
            
            # Goal statistics
            'goals_scored': [], 'goals_conceded': [], 'total_goals': [],
            'home_goals_scored': [], 'home_goals_conceded': [],
            'away_goals_scored': [], 'away_goals_conceded': [],
            
            # Advanced performance metrics
            'recent_form': deque(maxlen=10), 'form_trend': [],
            'performance_rating': [], 'consistency_score': [],
            'surprise_results': 0, 'expected_results': 0,
            
            # Market-based metrics
            'avg_odds_when_home': [], 'avg_odds_when_away': [],
            'odds_accuracy': [], 'value_bet_success': [],
            
            # Temporal patterns
            'weekend_performance': [], 'weekday_performance': [],
            'evening_performance': [], 'afternoon_performance': [],
            
            # League-specific
            'leagues': set(), 'league_performance': defaultdict(list),
            
            # Streaks and momentum
            'current_win_streak': 0, 'current_loss_streak': 0,
            'max_win_streak': 0, 'max_loss_streak': 0,
            'goal_scoring_streaks': [], 'clean_sheet_streaks': []
        })
        
        # Sort by time for proper sequence analysis
        df_sorted = df.sort_values('start_time') if 'start_time' in df.columns else df
        
        for _, match in df_sorted.iterrows():
            self.update_team_statistics(match)
        
        # Calculate final derived statistics
        self.calculate_derived_team_stats()
    
    def update_team_statistics(self, match):
        """Update team statistics with a single match"""
        home_team = match['home_team']
        away_team = match['away_team']
        league = match['league']
        
        # Extract match results if available
        if 'actual_home_score' in match and 'actual_away_score' in match:
            home_score = match['actual_home_score']
            away_score = match['actual_away_score']
            actual_winner = match.get('actual_winner', -1)
            
            # Update basic statistics
            for team, is_home, score_for, score_against in [
                (home_team, True, home_score, away_score),
                (away_team, False, away_score, home_score)
            ]:
                stats = self.team_stats[team]
                stats['total_matches'] += 1
                stats['goals_scored'].append(score_for)
                stats['goals_conceded'].append(score_against)
                stats['total_goals'].append(home_score + away_score)
                stats['leagues'].add(league)
                
                # Determine result for this team
                if is_home:
                    stats['home_matches'] += 1
                    stats['home_goals_scored'].append(score_for)
                    stats['home_goals_conceded'].append(score_against)
                    
                    if actual_winner == 2:  # Home win
                        result = 'W'
                        stats['wins'] += 1
                        stats['home_wins'] += 1
                    elif actual_winner == 1:  # Draw
                        result = 'D'
                        stats['draws'] += 1
                        stats['home_draws'] += 1
                    else:  # Away win
                        result = 'L'
                        stats['losses'] += 1
                        stats['home_losses'] += 1
                else:
                    stats['away_matches'] += 1
                    stats['away_goals_scored'].append(score_for)
                    stats['away_goals_conceded'].append(score_against)
                    
                    if actual_winner == 0:  # Away win
                        result = 'W'
                        stats['wins'] += 1
                        stats['away_wins'] += 1
                    elif actual_winner == 1:  # Draw
                        result = 'D'
                        stats['draws'] += 1
                        stats['away_draws'] += 1
                    else:  # Home win
                        result = 'L'
                        stats['losses'] += 1
                        stats['away_losses'] += 1
                
                # Update form and streaks
                stats['recent_form'].append(result)
                self.update_streaks(stats, result)
                
                # League-specific performance
                stats['league_performance'][league].append(result)
                
                # Temporal performance
                if 'is_weekend' in match:
                    if match['is_weekend']:
                        stats['weekend_performance'].append(result)
                    else:
                        stats['weekday_performance'].append(result)
                
                if 'is_evening' in match:
                    if match['is_evening']:
                        stats['evening_performance'].append(result)
                    else:
                        stats['afternoon_performance'].append(result)
    
    def update_streaks(self, stats, result):
        """Update win/loss streaks"""
        if result == 'W':
            stats['current_win_streak'] += 1
            stats['current_loss_streak'] = 0
            stats['max_win_streak'] = max(stats['max_win_streak'], stats['current_win_streak'])
        elif result == 'L':
            stats['current_loss_streak'] += 1
            stats['current_win_streak'] = 0
            stats['max_loss_streak'] = max(stats['max_loss_streak'], stats['current_loss_streak'])
        else:  # Draw
            stats['current_win_streak'] = 0
            stats['current_loss_streak'] = 0
    
    def calculate_derived_team_stats(self):
        """Calculate advanced derived statistics for all teams"""
        for team, stats in self.team_stats.items():
            # Basic rates
            if stats['total_matches'] > 0:
                stats['win_rate'] = stats['wins'] / stats['total_matches']
                stats['draw_rate'] = stats['draws'] / stats['total_matches']
                stats['loss_rate'] = stats['losses'] / stats['total_matches']
            
            # Home/Away specific rates
            if stats['home_matches'] > 0:
                stats['home_win_rate'] = stats['home_wins'] / stats['home_matches']
                stats['home_points_per_game'] = (stats['home_wins'] * 3 + stats['home_draws']) / stats['home_matches']
            
            if stats['away_matches'] > 0:
                stats['away_win_rate'] = stats['away_wins'] / stats['away_matches']
                stats['away_points_per_game'] = (stats['away_wins'] * 3 + stats['away_draws']) / stats['away_matches']
            
            # Goal statistics
            if stats['goals_scored']:
                stats['avg_goals_scored'] = np.mean(stats['goals_scored'])
                stats['avg_goals_conceded'] = np.mean(stats['goals_conceded'])
                stats['goal_difference'] = stats['avg_goals_scored'] - stats['avg_goals_conceded']
                stats['goals_scored_std'] = np.std(stats['goals_scored'])
                stats['goals_conceded_std'] = np.std(stats['goals_conceded'])
                
                # Attacking and defensive consistency
                stats['attacking_consistency'] = 1 / (1 + stats['goals_scored_std'])
                stats['defensive_consistency'] = 1 / (1 + stats['goals_conceded_std'])
            
            # Form analysis
            if stats['recent_form']:
                recent_results = list(stats['recent_form'])
                stats['recent_wins'] = recent_results.count('W')
                stats['recent_draws'] = recent_results.count('D')
                stats['recent_losses'] = recent_results.count('L')
                stats['recent_points'] = stats['recent_wins'] * 3 + stats['recent_draws']
                stats['form_momentum'] = self.calculate_form_momentum(recent_results)
            
            # Performance rating (comprehensive)
            stats['performance_rating'] = self.calculate_performance_rating(stats)
            
            # Consistency score
            stats['consistency_score'] = self.calculate_consistency_score(stats)
    
    def calculate_form_momentum(self, recent_results):
        """Calculate momentum based on recent form with weighted recency"""
        if not recent_results:
            return 0.0
        
        weights = np.exp(np.linspace(-1, 0, len(recent_results)))  # More weight to recent
        points = []
        
        for result in recent_results:
            if result == 'W':
                points.append(3)
            elif result == 'D':
                points.append(1)
            else:
                points.append(0)
        
        weighted_points = np.average(points, weights=weights)
        return weighted_points / 3.0  # Normalize to 0-1
    
    def calculate_performance_rating(self, stats):
        """Calculate comprehensive performance rating"""
        components = []
        
        # Win rate component (40%)
        if stats['total_matches'] > 0:
            components.append(('win_rate', stats['win_rate'], 0.4))
        
        # Goal difference component (25%)
        if 'goal_difference' in stats:
            # Normalize goal difference to 0-1 scale
            normalized_gd = 1 / (1 + np.exp(-stats['goal_difference']))
            components.append(('goal_diff', normalized_gd, 0.25))
        
        # Form momentum component (20%)
        if 'form_momentum' in stats:
            components.append(('momentum', stats['form_momentum'], 0.2))
        
        # Consistency component (15%)
        if 'consistency_score' in stats:
            components.append(('consistency', stats['consistency_score'], 0.15))
        
        # Calculate weighted average
        if components:
            total_weight = sum(weight for _, _, weight in components)
            weighted_sum = sum(value * weight for _, value, weight in components)
            return weighted_sum / total_weight
        
        return 0.5
    
    def calculate_consistency_score(self, stats):
        """Calculate team consistency score"""
        consistency_factors = []
        
        # Goal scoring consistency
        if 'attacking_consistency' in stats:
            consistency_factors.append(stats['attacking_consistency'])
        
        # Defensive consistency
        if 'defensive_consistency' in stats:
            consistency_factors.append(stats['defensive_consistency'])
        
        # Result consistency (less variance in results)
        if stats['total_matches'] > 5:
            win_rate = stats['win_rate']
            draw_rate = stats['draw_rate']
            loss_rate = stats['loss_rate']
            
            # Higher entropy = less consistent
            result_entropy = -sum(p * np.log(p + 1e-10) for p in [win_rate, draw_rate, loss_rate])
            result_consistency = 1 / (1 + result_entropy)
            consistency_factors.append(result_consistency)
        
        return np.mean(consistency_factors) if consistency_factors else 0.5
    
    def load_and_process_data(self):
        """Load and process all data with advanced preprocessing"""
        print("=== LOADING AND PROCESSING DATA ===")
        
        # Load training data (odds)
        self.training_data = self.load_training_data()
        if self.training_data.empty:
            print("No training data found")
            return False
        
        # Load results data
        self.results_data = self.load_results_data()
        if self.results_data.empty:
            print("No results data found")
            return False
        
        # Match odds with results
        self.matched_data = self.match_odds_with_results()
        if self.matched_data.empty:
            print("No matched data found")
            return False
        
        print(f"Successfully processed {len(self.matched_data)} matched records")
        
        # Build advanced statistics
        self.build_advanced_team_statistics(self.matched_data)
        
        # Analyze patterns
        patterns = self.pattern_recognizer.analyze_odds_patterns(self.matched_data)
        self.pattern_recognizer.build_decision_trees(self.matched_data)
        
        print("Pattern analysis completed")
        return True
    
    def load_training_data(self):
        """Load training data with enhanced processing"""
        all_matches = []
        
        if not os.path.exists(self.train_folder):
            return pd.DataFrame()
        
        json_files = [f for f in os.listdir(self.train_folder) if f.endswith('.json')]
        
        for filename in json_files:
            filepath = os.path.join(self.train_folder, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                for match in data.get('items', []):
                    features = self.extract_advanced_features(match)
                    features['source_file'] = filename
                    all_matches.append(features)
                    
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        
        return pd.DataFrame(all_matches)
    
    def load_results_data(self):
        """Load results data with enhanced processing"""
        all_results = []
        
        if not os.path.exists(self.results_folder):
            return pd.DataFrame()
        
        json_files = [f for f in os.listdir(self.results_folder) if f.endswith('.json')]
        
        for filename in json_files:
            filepath = os.path.join(self.results_folder, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                for match in data.get('items', []):
                    result_features = self.extract_match_result(match)
                    all_results.append(result_features)
                    
            except Exception as e:
                print(f"Error loading results from {filename}: {e}")
        
        return pd.DataFrame(all_results)
    
    def extract_match_result(self, result_data):
        """Extract match results with enhanced processing"""
        features = {}
        
        # Basic info
        features['match_id'] = result_data['id']
        features['home_team'] = result_data['participants'][0]['name']
        features['away_team'] = result_data['participants'][1]['name']
        features['league'] = result_data['competition']['name']
        features['region'] = result_data['region']['name']
        features['start_time'] = result_data['startTime']
        
        # Extract scores
        for participant_result in result_data['results']['participantPeriodResults']:
            participant_type = participant_result['participant']['type']
            
            for period_result in participant_result['periodResults']:
                if period_result['period']['slug'] == 'FULL_TIME_EXCLUDING_OVERTIME':
                    score = int(period_result['result'])
                    
                    if participant_type == 'HOME':
                        features['actual_home_score'] = score
                    elif participant_type == 'AWAY':
                        features['actual_away_score'] = score
        
        # Calculate derived results
        if 'actual_home_score' in features and 'actual_away_score' in features:
            home_score = features['actual_home_score']
            away_score = features['actual_away_score']
            
            # Winner
            if home_score > away_score:
                features['actual_winner'] = 2  # Home
            elif away_score > home_score:
                features['actual_winner'] = 0  # Away
            else:
                features['actual_winner'] = 1  # Draw
            
            # Goals
            features['actual_total_goals'] = home_score + away_score
            features['actual_btts'] = 1 if (home_score > 0 and away_score > 0) else 0
            
            # Over/Under
            for line in [1.5, 2.5, 3.5]:
                features[f'actual_over_{line}'] = 1 if features['actual_total_goals'] > line else 0
        
        return features
    
    def match_odds_with_results(self):
        """Advanced matching algorithm with fuzzy matching"""
        print("Matching odds with results using advanced algorithm...")
        
        matched_records = []
        
        # Create efficient lookup structures
        results_by_teams = defaultdict(list)
        for _, result in self.results_data.iterrows():
            key = f"{result['home_team']}_{result['away_team']}"
            results_by_teams[key].append(result)
        
        for _, odds_match in self.training_data.iterrows():
            key = f"{odds_match['home_team']}_{odds_match['away_team']}"
            
            if key in results_by_teams:
                # Find best time match
                best_match = self.find_best_time_match(odds_match, results_by_teams[key])
                
                if best_match is not None:
                    # Combine data
                    combined = odds_match.to_dict()
                    combined.update({
                        col: best_match[col] for col in best_match.index 
                        if col.startswith('actual_')
                    })
                    matched_records.append(combined)
        
        matched_df = pd.DataFrame(matched_records)
        print(f"Matched {len(matched_df)} records")
        return matched_df
    
    def find_best_time_match(self, odds_match, result_candidates):
        """Find best time match with tolerance"""
        try:
            odds_time = datetime.fromisoformat(odds_match['start_time'].replace('Z', '+00:00'))
            
            best_match = None
            min_time_diff = float('inf')
            
            for result_match in result_candidates:
                try:
                    result_time = datetime.fromisoformat(result_match['start_time'].replace('Z', '+00:00'))
                    time_diff = abs((odds_time - result_time).total_seconds())
                    
                    if time_diff < min_time_diff and time_diff <= 3600:  # 1 hour tolerance
                        min_time_diff = time_diff
                        best_match = result_match
                except:
                    continue
            
            return best_match
            
        except:
            return result_candidates[0] if result_candidates else None
    
    def prepare_advanced_features(self, df, is_training=False):
        """Advanced feature preparation with selection and engineering"""
        if df.empty:
            return df, []
        
        df_processed = df.copy()
        
        # 1. Advanced feature engineering
        df_processed = self.feature_engineer.create_advanced_features(df_processed)
        
        # 2. Add team-based features
        df_processed = self.add_advanced_team_features(df_processed)
        
        # 3. Encode categorical variables
        df_processed = self.encode_categorical_features(df_processed, is_training)
        
        # 4. Handle missing values intelligently
        df_processed = self.handle_missing_values(df_processed, is_training)
        
        # 5. Feature selection
        if is_training:
            df_processed, selected_features = self.select_optimal_features(df_processed)
            self.feature_columns = selected_features
        else:
            # Use previously selected features
            df_processed = df_processed[self.feature_columns]
        
        # 6. Feature scaling
        df_processed = self.scale_features(df_processed, is_training)
        
        return df_processed, self.feature_columns
    
    def add_advanced_team_features(self, df):
        """Add sophisticated team-based features"""
        for idx in df.index:
            home_team = df.loc[idx, 'home_team']
            away_team = df.loc[idx, 'away_team']
            league = df.loc[idx, 'league']
            
            # Home team features
            home_stats = self.team_stats.get(home_team, {})
            self.add_team_features_to_row(df, idx, home_stats, 'home')
            
            # Away team features
            away_stats = self.team_stats.get(away_team, {})
            self.add_team_features_to_row(df, idx, away_stats, 'away')
            
            # Comparative features
            self.add_comparative_features(df, idx, home_stats, away_stats)
            
            # Head-to-head features
            self.add_h2h_features(df, idx, home_team, away_team)
            
            # League-specific features
            self.add_league_features(df, idx, league)
        
        return df
    
    def add_team_features_to_row(self, df, idx, stats, prefix):
        """Add team features with specified prefix"""
        feature_mapping = {
            f'{prefix}_performance_rating': stats.get('performance_rating', 0.5),
            f'{prefix}_consistency_score': stats.get('consistency_score', 0.5),
            f'{prefix}_win_rate': stats.get('win_rate', 0.33),
            f'{prefix}_form_momentum': stats.get('form_momentum', 0.5),
            f'{prefix}_avg_goals_scored': stats.get('avg_goals_scored', 1.5),
            f'{prefix}_avg_goals_conceded': stats.get('avg_goals_conceded', 1.5),
            f'{prefix}_goal_difference': stats.get('goal_difference', 0),
            f'{prefix}_attacking_consistency': stats.get('attacking_consistency', 0.5),
            f'{prefix}_defensive_consistency': stats.get('defensive_consistency', 0.5),
            f'{prefix}_current_win_streak': stats.get('current_win_streak', 0),
            f'{prefix}_current_loss_streak': stats.get('current_loss_streak', 0),
            f'{prefix}_max_win_streak': stats.get('max_win_streak', 0),
        }
        
        # Home/Away specific features
        if prefix == 'home':
            feature_mapping.update({
                f'{prefix}_home_win_rate': stats.get('home_win_rate', 0.33),
                f'{prefix}_home_points_per_game': stats.get('home_points_per_game', 1.5),
            })
        else:
            feature_mapping.update({
                f'{prefix}_away_win_rate': stats.get('away_win_rate', 0.33),
                f'{prefix}_away_points_per_game': stats.get('away_points_per_game', 1.5),
            })
        
        for feature_name, value in feature_mapping.items():
            df.loc[idx, feature_name] = value
    
    def add_comparative_features(self, df, idx, home_stats, away_stats):
        """Add comparative features between teams"""
        # Performance comparison
        home_rating = home_stats.get('performance_rating', 0.5)
        away_rating = away_stats.get('performance_rating', 0.5)
        df.loc[idx, 'performance_rating_diff'] = home_rating - away_rating
        df.loc[idx, 'performance_rating_ratio'] = home_rating / (away_rating + 1e-10)
        
        # Form comparison
        home_momentum = home_stats.get('form_momentum', 0.5)
        away_momentum = away_stats.get('form_momentum', 0.5)
        df.loc[idx, 'momentum_diff'] = home_momentum - away_momentum
        
        # Goal statistics comparison
        home_goals_scored = home_stats.get('avg_goals_scored', 1.5)
        away_goals_scored = away_stats.get('avg_goals_scored', 1.5)
        home_goals_conceded = home_stats.get('avg_goals_conceded', 1.5)
        away_goals_conceded = away_stats.get('avg_goals_conceded', 1.5)
        
        df.loc[idx, 'attacking_strength_diff'] = home_goals_scored - away_goals_scored
        df.loc[idx, 'defensive_strength_diff'] = away_goals_conceded - home_goals_conceded
        
        # Expected goals calculation
        df.loc[idx, 'expected_home_goals'] = (home_goals_scored + away_goals_conceded) / 2
        df.loc[idx, 'expected_away_goals'] = (away_goals_scored + home_goals_conceded) / 2
        df.loc[idx, 'expected_total_goals'] = df.loc[idx, 'expected_home_goals'] + df.loc[idx, 'expected_away_goals']
        
        # Consistency comparison
        home_consistency = home_stats.get('consistency_score', 0.5)
        away_consistency = away_stats.get('consistency_score', 0.5)
        df.loc[idx, 'consistency_diff'] = home_consistency - away_consistency
    
    def add_h2h_features(self, df, idx, home_team, away_team):
        """Add head-to-head features"""
        h2h_key = f"{home_team}_vs_{away_team}"
        reverse_key = f"{away_team}_vs_{home_team}"
        
        h2h_stats = self.h2h_stats.get(h2h_key, self.h2h_stats.get(reverse_key, {}))
        
        df.loc[idx, 'h2h_matches'] = h2h_stats.get('total_matches', 0)
        df.loc[idx, 'h2h_home_advantage'] = h2h_stats.get('home_win_rate', 0.33)
        df.loc[idx, 'h2h_avg_goals'] = h2h_stats.get('avg_total_goals', 2.5)
        df.loc[idx, 'h2h_btts_rate'] = h2h_stats.get('btts_rate', 0.5)
    
    def add_league_features(self, df, idx, league):
        """Add league-specific features"""
        league_stats = self.league_stats.get(league, {})
        
        df.loc[idx, 'league_avg_goals'] = league_stats.get('avg_goals_per_match', 2.5)
        df.loc[idx, 'league_home_advantage'] = league_stats.get('home_win_rate', 0.45)
        df.loc[idx, 'league_draw_rate'] = league_stats.get('draw_rate', 0.25)
        df.loc[idx, 'league_btts_rate'] = league_stats.get('btts_rate', 0.5)
    
    def encode_categorical_features(self, df, is_training):
        """Advanced categorical encoding"""
        categorical_cols = ['home_team', 'away_team', 'league', 'region']
        
        for col in categorical_cols:
            if col in df.columns:
                if is_training:
                    self.encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.encoders[col].fit_transform(df[col].astype(str))
                else:
                    if col in self.encoders:
                        # Handle unseen categories
                        known_categories = set(self.encoders[col].classes_)
                        df_categories = set(df[col].astype(str).unique())
                        unknown_categories = df_categories - known_categories
                        
                        if unknown_categories:
                            # Assign unknown categories to most frequent class
                            most_frequent = self.encoders[col].classes_[0]
                            df[col] = df[col].astype(str).replace(list(unknown_categories), most_frequent)
                        
                        df[f'{col}_encoded'] = self.encoders[col].transform(df[col].astype(str))
                    else:
                        df[f'{col}_encoded'] = 0
        
        return df
    
    def handle_missing_values(self, df, is_training):
        """Intelligent missing value handling"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if is_training:
                if col.endswith('_odds'):
                    fill_value = 2.0  # Default odds
                elif col.endswith('_prob'):
                    fill_value = 0.5  # Default probability
                elif 'goals' in col:
                    fill_value = 1.5  # Default goals
                elif 'rate' in col or 'percentage' in col:
                    fill_value = 0.33  # Default rate
                else:
                    fill_value = df[col].median()
                
                self.fill_values[col] = fill_value
            else:
                fill_value = self.fill_values.get(col, 0)
            
            df[col] = df[col].fillna(fill_value)
        
        return df
    
    def select_optimal_features(self, df):
        """Advanced feature selection using multiple methods"""
        # Exclude target and non-feature columns
        exclude_cols = [
            'home_team', 'away_team', 'league', 'region', 'match_id', 'start_time',
            'source_file', 'start_datetime'
        ] + [col for col in df.columns if col.startswith('actual_')]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        if 'actual_winner' in df.columns:
            X = df[feature_cols]
            y = df['actual_winner']
            
            # Remove low variance features
            variance_threshold = 0.01
            high_variance_cols = [col for col in feature_cols if X[col].var() > variance_threshold]
            
            # Statistical feature selection
            if len(high_variance_cols) > 50:  # Only if we have many features
                selector = SelectKBest(score_func=f_classif, k=min(50, len(high_variance_cols)))
                X_selected = selector.fit_transform(X[high_variance_cols], y)
                selected_features = [high_variance_cols[i] for i in selector.get_support(indices=True)]
            else:
                selected_features = high_variance_cols
            
            print(f"Selected {len(selected_features)} features from {len(feature_cols)} total")
            return df[selected_features + [col for col in exclude_cols if col in df.columns]], selected_features
        
        return df, feature_cols
    
    def scale_features(self, df, is_training):
        """Advanced feature scaling"""
        if is_training:
            self.scalers['robust'] = RobustScaler()
            scaled_data = self.scalers['robust'].fit_transform(df)
        else:
            if 'robust' in self.scalers:
                scaled_data = self.scalers['robust'].transform(df)
            else:
                scaled_data = df.values
        
        return pd.DataFrame(scaled_data, columns=df.columns, index=df.index)
    
    def create_advanced_targets(self, df):
        """Create targets with confidence weighting"""
        targets = {}
        
        if 'actual_winner' in df.columns:
            targets['winner'] = df['actual_winner'].values
            
            # Create confidence weights based on odds accuracy
            if all(col in df.columns for col in ['home_win_odds', 'away_win_odds', 'draw_odds']):
                confidence_weights = []
                for _, match in df.iterrows():
                    odds = [match['away_win_odds'], match['draw_odds'], match['home_win_odds']]
                    favorite = np.argmin(odds)
                    actual = match['actual_winner']
                    
                    # Higher confidence for matches where favorite won
                    if favorite == actual:
                        weight = 1.0 + (1.0 / odds[favorite])  # Higher weight for lower odds
                    else:
                        weight = 0.5  # Lower weight for upsets
                    
                    confidence_weights.append(weight)
                
                targets['winner_weights'] = np.array(confidence_weights)
        
        # Over/Under targets
        for line in [1.5, 2.5, 3.5]:
            actual_col = f'actual_over_{line}'
            if actual_col in df.columns:
                targets[f'over_under_{line}'] = df[actual_col].values
        
        # BTTS target
        if 'actual_btts' in df.columns:
            targets['btts'] = df['actual_btts'].values
        
        return targets
    
    def train_advanced_models(self):
        """Train advanced models with ensemble techniques"""
        print("=== TRAINING ADVANCED MODELS ===")
        
        if not self.load_and_process_data():
            return
        
        # Prepare features
        X, feature_names = self.prepare_advanced_features(self.matched_data, is_training=True)
        
        if X.empty:
            print("No features available for training")
            return
        
        # Create targets
        targets = self.create_advanced_targets(self.matched_data)
        
        # Train models for each target
        for target_name, y in targets.items():
            if target_name.endswith('_weights'):
                continue
            
            print(f"\nTraining advanced models for {target_name}...")
            
            # Get sample weights if available
            weights = targets.get(f'{target_name}_weights', None)
            
            # Train ensemble of models
            ensemble_model = self.train_ensemble_model(X, y, target_name, weights)
            
            if ensemble_model:
                self.models[target_name] = ensemble_model
                
                # Evaluate model
                self.evaluate_model_performance(X, y, ensemble_model, target_name)
        
        print(f"\nTrained {len(self.models)} advanced models")
    
    def train_ensemble_model(self, X, y, target_name, sample_weights=None):
        """Train advanced ensemble model"""
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        if sample_weights is not None:
            weights_train = sample_weights[:len(X_train)]
            weights_test = sample_weights[len(X_train):]
        else:
            weights_train = None
            weights_test = None
        
        # Create base models
        base_models = self.ensemble_predictor.create_base_models()
        
        # Train and evaluate base models
        trained_models = {}
        model_scores = {}
        
        for model_name, model in base_models.items():
            try:
                # Hyperparameter optimization for key models
                if model_name in ['xgboost', 'random_forest'] and OPTUNA_AVAILABLE:
                    model = self.ensemble_predictor.optimize_hyperparameters(X_train, y_train, model_name, model)
                
                # Train model
                if sample_weights is not None and hasattr(model, 'fit') and 'sample_weight' in model.fit.__code__.co_varnames:
                    model.fit(X_train, y_train, sample_weight=weights_train)
                else:
                    model.fit(X_train, y_train)
                
                # Evaluate
                train_score = model.score(X_train, y_train)
                test_score = model.score(X_test, y_test)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
                
                model_scores[model_name] = {
                    'train': train_score,
                    'test': test_score,
                    'cv_mean': cv_mean,
                    'cv_std': cv_std,
                    'stability': cv_mean - cv_std
                }
                
                trained_models[model_name] = model
                
                print(f"  {model_name}: CV {cv_mean:.3f}±{cv_std:.3f}, Test {test_score:.3f}")
                
            except Exception as e:
                print(f"  Error training {model_name}: {e}")
        
        if not trained_models:
            return None
        
        # Select best models for ensemble (top 5)
        best_models = sorted(model_scores.items(), key=lambda x: x[1]['stability'], reverse=True)[:5]
        ensemble_models = {name: trained_models[name] for name, _ in best_models}
        
        # Create ensemble
        if len(ensemble_models) >= 3:
            # Stacking ensemble
            ensemble = self.ensemble_predictor.create_stacked_ensemble(ensemble_models, X_train, y_train)
            ensemble.fit(X_train, y_train)
            
            ensemble_score = ensemble.score(X_test, y_test)
            print(f"  Ensemble score: {ensemble_score:.3f}")
            
            return ensemble
        else:
            # Return best single model
            best_model_name = best_models[0][0]
            return trained_models[best_model_name]
    
    def evaluate_model_performance(self, X, y, model, target_name):
        """Comprehensive model evaluation"""
        try:
            # Predictions
            y_pred = model.predict(X)
            
            # Basic metrics
            accuracy = accuracy_score(y, y_pred)
            self.prediction_accuracy[target_name].append(accuracy)
            
            # Detailed evaluation for winner prediction
            if target_name == 'winner':
                # Classification report
                report = classification_report(y, y_pred, output_dict=True)
                
                # Confusion matrix analysis
                cm = confusion_matrix(y, y_pred)
                
                # Calculate per-class accuracy
                class_accuracies = cm.diagonal() / cm.sum(axis=1)
                
                print(f"    Overall Accuracy: {accuracy:.3f}")
                print(f"    Away Win Accuracy: {class_accuracies[0]:.3f}")
                print(f"    Draw Accuracy: {class_accuracies[1]:.3f}")
                print(f"    Home Win Accuracy: {class_accuracies[2]:.3f}")
                
                # Probability calibration
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X)
                    log_loss_score = log_loss(y, y_proba)
                    print(f"    Log Loss: {log_loss_score:.3f}")
            
            # Feature importance if available
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': self.feature_columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                self.feature_importance_history[target_name].append(importance_df)
                
                print(f"    Top 3 features:")
                for _, row in importance_df.head(3).iterrows():
                    print(f"      {row['feature']}: {row['importance']:.3f}")
        
        except Exception as e:
            print(f"    Error in evaluation: {e}")
    
    def predict_with_advanced_analysis(self, home_team, away_team, league, upcoming_filename=None):
        """Advanced prediction with comprehensive analysis"""
        if not self.models:
            print("No trained models available")
            return None
        
        print(f"\n=== ADVANCED PREDICTION ANALYSIS ===")
        print(f"Match: {home_team} vs {away_team}")
        print(f"League: {league}")
        
        # Load upcoming data if provided
        match_data = None
        if upcoming_filename:
            upcoming_df = self.load_upcoming_data(upcoming_filename)
            if not upcoming_df.empty:
                match_data = self.find_match_in_upcoming(home_team, away_team, league, upcoming_df)
        
        if match_data is None:
            # Create synthetic match data
            match_data = self.create_synthetic_match_data(home_team, away_team, league)
        
        # Prepare features
        match_df = pd.DataFrame([match_data])
        X, _ = self.prepare_advanced_features(match_df, is_training=False)
        
        if X.empty:
            print("Could not prepare features for prediction")
            return None
        
        # Make predictions
        predictions = {}
        probabilities = {}
        confidence_scores = {}
        
        for target_name, model in self.models.items():
            try:
                pred = model.predict(X)[0]
                
                if hasattr(model, 'predict_proba'):
                    pred_proba = model.predict_proba(X)[0]
                    confidence = np.max(pred_proba)
                    
                    # Adjust confidence based on historical performance
                    if target_name in self.prediction_accuracy:
                        historical_accuracy = np.mean(self.prediction_accuracy[target_name])
                        adjusted_confidence = confidence * historical_accuracy
                    else:
                        adjusted_confidence = confidence
                else:
                    pred_proba = None
                    adjusted_confidence = 0.5
                
                predictions[target_name] = pred
                probabilities[target_name] = pred_proba
                confidence_scores[target_name] = adjusted_confidence
                
            except Exception as e:
                print(f"Error predicting {target_name}: {e}")
        
        # Advanced analysis
        analysis = self.perform_advanced_match_analysis(home_team, away_team, league, match_data)
        
        # Pattern-based insights
        pattern_insights = self.get_pattern_insights(match_data)
        
        # Display results
        self.display_advanced_prediction_results(
            predictions, probabilities, confidence_scores, 
            analysis, pattern_insights, match_data
        )
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'confidence': confidence_scores,
            'analysis': analysis,
            'pattern_insights': pattern_insights,
            'match_data': match_data
        }
    
    def load_upcoming_data(self, filename):
        """Load upcoming match data"""
        filepath = os.path.join(self.train_folder, filename)
        
        if not os.path.exists(filepath):
            return pd.DataFrame()
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            matches = []
            for match in data.get('items', []):
                features = self.extract_advanced_features(match)
                matches.append(features)
            
            return pd.DataFrame(matches)
            
        except Exception as e:
            print(f"Error loading upcoming data: {e}")
            return pd.DataFrame()
    
    def find_match_in_upcoming(self, home_team, away_team, league, upcoming_df):
        """Find specific match in upcoming data"""
        # Exact match
        exact_match = upcoming_df[
            (upcoming_df['home_team'].str.upper() == home_team.upper()) &
            (upcoming_df['away_team'].str.upper() == away_team.upper()) &
            (upcoming_df['league'].str.upper() == league.upper())
        ]
        
        if not exact_match.empty:
            return exact_match.iloc[0].to_dict()
        
        # Fuzzy match
        for _, match in upcoming_df.iterrows():
            if (self.fuzzy_match(match['home_team'], home_team) and
                self.fuzzy_match(match['away_team'], away_team) and
                self.fuzzy_match(match['league'], league)):
                return match.to_dict()
        
        return None
    
    def fuzzy_match(self, str1, str2, threshold=0.8):
        """Simple fuzzy string matching"""
        str1_clean = str1.lower().strip()
        str2_clean = str2.lower().strip()
        
        # Check if one string contains the other
        if str1_clean in str2_clean or str2_clean in str1_clean:
            return True
        
        # Check character overlap
        set1 = set(str1_clean)
        set2 = set(str2_clean)
        overlap = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return (overlap / union) >= threshold if union > 0 else False
    
    def create_synthetic_match_data(self, home_team, away_team, league):
        """Create synthetic match data for prediction"""
        # Get team statistics
        home_stats = self.team_stats.get(home_team, {})
        away_stats = self.team_stats.get(away_team, {})
        
        # Estimate odds based on team performance
        home_rating = home_stats.get('performance_rating', 0.5)
        away_rating = away_stats.get('performance_rating', 0.5)
        
        # Simple odds estimation
        if home_rating > away_rating:
            home_odds = 1.5 + (away_rating / home_rating) * 2
            away_odds = 2.0 + (home_rating / away_rating) * 3
        else:
            home_odds = 2.0 + (away_rating / home_rating) * 3
            away_odds = 1.5 + (home_rating / away_rating) * 2
        
        draw_odds = 3.0 + abs(home_rating - away_rating) * 2
        
        # Create match data
        now = datetime.now()
        match_data = {
            'home_team': home_team,
            'away_team': away_team,
            'league': league,
            'region': 'Unknown',
            'match_id': f'SYNTHETIC_{home_team}_{away_team}',
            'start_time': now.isoformat(),
            'home_win_odds': home_odds,
            'away_win_odds': away_odds,
            'draw_odds': draw_odds,
            'home_win_prob': 1/home_odds,
            'away_win_prob': 1/away_odds,
            'draw_prob': 1/draw_odds,
        }
        
        # Add temporal features
        match_data.update(self.extract_temporal_features(now))
        
        # Add derived features
        derived = self.calculate_advanced_derived_features(match_data)
        match_data.update(derived)
        
        return match_data
    
    def perform_advanced_match_analysis(self, home_team, away_team, league, match_data):
        """Perform comprehensive match analysis"""
        analysis = {}
        
        # Team performance analysis
        home_stats = self.team_stats.get(home_team, {})
        away_stats = self.team_stats.get(away_team, {})
        
        analysis['team_comparison'] = {
            'home_performance_rating': home_stats.get('performance_rating', 0.5),
            'away_performance_rating': away_stats.get('performance_rating', 0.5),
            'performance_advantage': home_stats.get('performance_rating', 0.5) - away_stats.get('performance_rating', 0.5),
            'home_form_momentum': home_stats.get('form_momentum', 0.5),
            'away_form_momentum': away_stats.get('form_momentum', 0.5),
            'momentum_advantage': home_stats.get('form_momentum', 0.5) - away_stats.get('form_momentum', 0.5)
        }
        
        # Goal expectation analysis
        expected_home_goals = match_data.get('expected_home_goals', 1.5)
        expected_away_goals = match_data.get('expected_away_goals', 1.5)
        
        analysis['goal_analysis'] = {
            'expected_home_goals': expected_home_goals,
            'expected_away_goals': expected_away_goals,
            'expected_total_goals': expected_home_goals + expected_away_goals,
            'expected_goal_difference': expected_home_goals - expected_away_goals,
            'high_scoring_probability': 1 / (1 + np.exp(-(expected_home_goals + expected_away_goals - 2.5)))
        }
        
        # Market analysis
        if all(k in match_data for k in ['home_win_odds', 'away_win_odds', 'draw_odds']):
            analysis['market_analysis'] = {
                'market_efficiency': match_data.get('market_efficiency', 0.5),
                'bookmaker_margin': match_data.get('bookmaker_margin', 0.1),
                'favorite': self.get_market_favorite(match_data),
                'favorite_strength': match_data.get('favorite_strength', 1.5),
                'draw_winner_diff': match_data.get('draw_winner_diff', 0),
                'high_draw_diff_pattern': match_data.get('high_draw_diff_pattern', 0)
            }
        
        return analysis
    
    def get_market_favorite(self, match_data):
        """Determine market favorite"""
        if all(k in match_data for k in ['home_win_odds', 'away_win_odds', 'draw_odds']):
            odds = [match_data['away_win_odds'], match_data['draw_odds'], match_data['home_win_odds']]
            favorite_idx = np.argmin(odds)
            return ['Away', 'Draw', 'Home'][favorite_idx]
        return 'Unknown'
    
    def get_pattern_insights(self, match_data):
        """Get insights from pattern recognition"""
        insights = {}
        
        # Your specific pattern analysis
        draw_winner_diff = match_data.get('draw_winner_diff', 0)
        if draw_winner_diff > 2.0:
            # Check historical accuracy of this pattern
            pattern_name = 'high_draw_diff'
            if pattern_name in self.pattern_recognizer.pattern_accuracy:
                historical_accuracy = np.mean(self.pattern_recognizer.pattern_accuracy[pattern_name])
                insights['high_draw_diff_pattern'] = {
                    'triggered': True,
                    'historical_accuracy': historical_accuracy,
                    'recommendation': 'Consider non-draw outcomes',
                    'confidence_boost': 0.1 if historical_accuracy > 0.6 else 0
                }
        
        # Market efficiency insights
        market_efficiency = match_data.get('market_efficiency', 0.5)
        if market_efficiency < 0.3:
            insights['low_efficiency_market'] = {
                'detected': True,
                'recommendation': 'High margin market - be cautious',
                'confidence_penalty': 0.1
            }
        
        return insights
    
    def display_advanced_prediction_results(self, predictions, probabilities, confidence_scores, analysis, pattern_insights, match_data):
        """Display comprehensive prediction results"""
        print(f"\n{'='*60}")
        print(f"ADVANCED PREDICTION RESULTS")
        print(f"{'='*60}")
        
        # Winner prediction
        if 'winner' in predictions:
            winner_pred = predictions['winner']
            winner_map = {0: match_data['away_team'], 1: 'Draw', 2: match_data['home_team']}
            predicted_winner = winner_map.get(winner_pred, 'Unknown')
            confidence = confidence_scores.get('winner', 0)
            
            print(f"\n🏆 MATCH WINNER PREDICTION:")
            print(f"   Predicted: {predicted_winner}")
            print(f"   Confidence: {confidence:.1%}")
            
            if probabilities['winner'] is not None:
                prob_dist = probabilities['winner']
                print(f"   Probability Distribution:")
                print(f"     {match_data['away_team']}: {prob_dist[0]:.1%}")
                print(f"     Draw: {prob_dist[1]:.1%}")
                print(f"     {match_data['home_team']}: {prob_dist[2]:.1%}")
        
        # Goals prediction
        print(f"\n⚽ GOALS ANALYSIS:")
        expected_total = analysis['goal_analysis']['expected_total_goals']
        print(f"   Expected Total Goals: {expected_total:.2f}")
        
        for line in [1.5, 2.5, 3.5]:
            target_name = f'over_under_{line}'
            if target_name in predictions:
                pred = predictions[target_name]
                confidence = confidence_scores.get(target_name, 0)
                result = 'Over' if pred == 1 else 'Under'
                print(f"   {line} Goals: {result} {line} (Confidence: {confidence:.1%})")
        
        # BTTS prediction
        if 'btts' in predictions:
            btts_pred = predictions['btts']
            btts_confidence = confidence_scores.get('btts', 0)
            btts_result = 'Yes' if btts_pred == 1 else 'No'
            print(f"\n🎯 BOTH TEAMS TO SCORE: {btts_result} (Confidence: {btts_confidence:.1%})")
        
        # Team analysis
        print(f"\n📊 TEAM ANALYSIS:")
        team_comp = analysis['team_comparison']
        print(f"   Performance Advantage: {team_comp['performance_advantage']:+.3f}")
        print(f"   Momentum Advantage: {team_comp['momentum_advantage']:+.3f}")
        
        # Market analysis
        if 'market_analysis' in analysis:
            market = analysis['market_analysis']
            print(f"\n💰 MARKET ANALYSIS:")
            print(f"   Market Favorite: {market['favorite']}")
            print(f"   Market Efficiency: {market['market_efficiency']:.1%}")
            print(f"   Bookmaker Margin: {market['bookmaker_margin']:.1%}")
            
            # Your specific pattern
            if market['high_draw_diff_pattern']:
                print(f"   🔍 HIGH DRAW DIFFERENCE PATTERN DETECTED!")
                print(f"       Draw vs Winner Diff: {market['draw_winner_diff']:.2f}")
                print(f"       Historical Pattern: Favors non-draw outcomes")
        
        # Pattern insights
        if pattern_insights:
            print(f"\n🧠 PATTERN INSIGHTS:")
            for pattern_name, insight in pattern_insights.items():
                if insight.get('triggered', False):
                    print(f"   {pattern_name}: {insight.get('recommendation', 'No recommendation')}")
                    if 'confidence_boost' in insight:
                        print(f"     Confidence adjustment: +{insight['confidence_boost']:.1%}")
        
        # Recommendation system
        self.provide_advanced_recommendations(predictions, confidence_scores, analysis, pattern_insights)
    
    def provide_advanced_recommendations(self, predictions, confidence_scores, analysis, pattern_insights):
        """Provide sophisticated betting recommendations"""
        print(f"\n🎲 ADVANCED RECOMMENDATIONS:")
        
        recommendations = []
        
        # Winner recommendation
        if 'winner' in predictions and 'winner' in confidence_scores:
            confidence = confidence_scores['winner']
            
            if confidence > 0.80:
                recommendations.append(f"STRONG BET: Winner prediction with {confidence:.1%} confidence")
            elif confidence > 0.70:
                recommendations.append(f"GOOD BET: Winner prediction with {confidence:.1%} confidence")
            elif confidence > 0.60:
                recommendations.append(f"MODERATE BET: Winner prediction with {confidence:.1%} confidence")
            else:
                recommendations.append(f"AVOID: Low confidence winner prediction ({confidence:.1%})")
        
        # Pattern-based recommendations
        for pattern_name, insight in pattern_insights.items():
            if insight.get('triggered', False) and 'recommendation' in insight:
                recommendations.append(f"PATTERN: {insight['recommendation']}")
        
        # Market efficiency recommendations
        if 'market_analysis' in analysis:
            market_eff = analysis['market_analysis']['market_efficiency']
            if market_eff < 0.3:
                recommendations.append("CAUTION: Low market efficiency - high bookmaker margin")
            elif market_eff > 0.8:
                recommendations.append("OPPORTUNITY: High market efficiency - competitive odds")
        
        # Display recommendations
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
        
        if not recommendations:
            print("   No specific recommendations - proceed with caution")
    
    def save_advanced_models(self, filename='advanced_sports_predictor.pkl'):
        """Save all models and data"""
        model_data = {
            'models': self.models,
            'ensemble_models': self.ensemble_models,
            'encoders': self.encoders,
            'scalers': self.scalers,
            'feature_selectors': self.feature_selectors,
            'feature_columns': self.feature_columns,
            'fill_values': self.fill_values,
            'team_stats': dict(self.team_stats),
            'h2h_stats': dict(self.h2h_stats),
            'league_stats': dict(self.league_stats),
            'prediction_accuracy': dict(self.prediction_accuracy),
            'pattern_recognizer': self.pattern_recognizer,
            'confidence_threshold': self.confidence_threshold,
            'ensemble_weights': self.ensemble_weights
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Advanced models saved to {filename}")
    
    def load_advanced_models(self, filename='advanced_sports_predictor.pkl'):
        """Load all models and data"""
        try:
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)
            
            self.models = model_data.get('models', {})
            self.ensemble_models = model_data.get('ensemble_models', {})
            self.encoders = model_data.get('encoders', {})
            self.scalers = model_data.get('scalers', {})
            self.feature_selectors = model_data.get('feature_selectors', {})
            self.feature_columns = model_data.get('feature_columns', [])
            self.fill_values = model_data.get('fill_values', {})
            self.team_stats = defaultdict(dict, model_data.get('team_stats', {}))
            self.h2h_stats = defaultdict(dict, model_data.get('h2h_stats', {}))
            self.league_stats = defaultdict(dict, model_data.get('league_stats', {}))
            self.prediction_accuracy = defaultdict(list, model_data.get('prediction_accuracy', {}))
            self.pattern_recognizer = model_data.get('pattern_recognizer', AdvancedPatternRecognition())
            self.confidence_threshold = model_data.get('confidence_threshold', 0.65)
            self.ensemble_weights = model_data.get('ensemble_weights', {})
            
            print(f"Advanced models loaded successfully")
            print(f"Models available: {list(self.models.keys())}")
            return True
            
        except FileNotFoundError:
            print(f"Model file {filename} not found")
            return False
        except Exception as e:
            print(f"Error loading models: {e}")
            return False

def main():
    """Main execution with advanced interface"""
    predictor = AdvancedSportsPredictor()
    
    # Try to load existing models
    if not predictor.load_advanced_models():
        print("Training new advanced models...")
        predictor.train_advanced_models()
        if predictor.models:
            predictor.save_advanced_models()
    
    print("\n=== ADVANCED SPORTS PREDICTION SYSTEM ===")
    print("Enhanced with:")
    print("- Advanced ensemble learning")
    print("- Pattern recognition algorithms")
    print("- Hyperparameter optimization")
    print("- Market efficiency analysis")
    print("- Your specific draw-difference pattern detection")
    print("\nCommands:")
    print("1. predict <home> <away> <league> [upcoming_file] - Advanced prediction")
    print("2. predict_all <upcoming_file> - Predict all upcoming matches")
    print("3. retrain - Retrain with latest data")
    print("4. analyze_patterns - Show pattern analysis")
    print("5. team_analysis <team> - Detailed team analysis")
    print("6. performance - Show model performance")
    print("7. save - Save models")
    print("8. quit")
    
    while True:
        try:
            command = input("\nEnter command: ").strip()
            
            if not command:
                continue
            
            parts = command.split()
            
            if parts[0] == 'quit':
                break
            elif parts[0] == 'predict' and len(parts) >= 4:
                home_team = parts[1]
                away_team = parts[2]
                league = parts[3]
                upcoming_file = parts[4] if len(parts) > 4 else None
                
                predictor.predict_with_advanced_analysis(home_team, away_team, league, upcoming_file)
                
            elif parts[0] == 'predict_all' and len(parts) >= 2:
                upcoming_file = parts[1]
                predictions = predictor.predict_all_upcoming_matches(upcoming_file)
                
            elif parts[0] == 'retrain':
                predictor.train_advanced_models()
                if predictor.models:
                    predictor.save_advanced_models()
                    
            elif parts[0] == 'save':
                predictor.save_advanced_models()
                
            else:
                print("Invalid command. Please try again.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()