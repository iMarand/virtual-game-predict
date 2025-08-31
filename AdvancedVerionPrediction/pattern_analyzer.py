import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

class AdvancedPatternAnalyzer:
    """Advanced pattern analyzer for discovering hidden betting patterns"""
    
    def __init__(self):
        self.discovered_patterns = {}
        self.pattern_performance = defaultdict(list)
        self.decision_rules = []
        self.clustering_models = {}
        
    def discover_draw_difference_patterns(self, df):
        """Discover and analyze your specific draw-difference pattern"""
        print("🔍 ANALYZING DRAW-DIFFERENCE PATTERNS...")
        
        if not all(col in df.columns for col in ['home_win_odds', 'away_win_odds', 'draw_odds', 'actual_winner']):
            return {}
        
        patterns = {}
        
        # Calculate draw vs winner differences
        df['min_winner_odds'] = df[['home_win_odds', 'away_win_odds']].min(axis=1)
        df['draw_winner_diff'] = df['draw_odds'] - df['min_winner_odds']
        
        # Analyze different thresholds
        thresholds = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        
        for threshold in thresholds:
            mask = df['draw_winner_diff'] > threshold
            if mask.sum() < 10:  # Need minimum samples
                continue
            
            subset = df[mask]
            total_matches = len(subset)
            
            # Analyze outcomes
            outcome_counts = subset['actual_winner'].value_counts()
            away_wins = outcome_counts.get(0, 0)
            draws = outcome_counts.get(1, 0)
            home_wins = outcome_counts.get(2, 0)
            
            non_draw_rate = (away_wins + home_wins) / total_matches
            home_win_rate = home_wins / total_matches
            away_win_rate = away_wins / total_matches
            draw_rate = draws / total_matches
            
            # Calculate pattern strength
            expected_non_draw_rate = 0.75  # Baseline expectation
            pattern_strength = non_draw_rate - expected_non_draw_rate
            
            patterns[f'threshold_{threshold}'] = {
                'total_matches': total_matches,
                'non_draw_rate': non_draw_rate,
                'home_win_rate': home_win_rate,
                'away_win_rate': away_win_rate,
                'draw_rate': draw_rate,
                'pattern_strength': pattern_strength,
                'statistical_significance': self.calculate_significance(non_draw_rate, total_matches, expected_non_draw_rate)
            }
            
            print(f"   Threshold {threshold}: {total_matches} matches, {non_draw_rate:.1%} non-draw rate")
        
        # Find optimal threshold
        best_threshold = self.find_optimal_threshold(patterns)
        patterns['optimal_threshold'] = best_threshold
        
        return patterns
    
    def calculate_significance(self, observed_rate, sample_size, expected_rate):
        """Calculate statistical significance of pattern"""
        if sample_size < 10:
            return 0.0
        
        # Simple z-test approximation
        std_error = np.sqrt(expected_rate * (1 - expected_rate) / sample_size)
        z_score = abs(observed_rate - expected_rate) / std_error
        
        # Convert to significance score (0-1)
        return min(1.0, z_score / 3.0)
    
    def find_optimal_threshold(self, patterns):
        """Find optimal threshold for draw-difference pattern"""
        best_threshold = 2.0
        best_score = 0
        
        for threshold_key, pattern_data in patterns.items():
            if not threshold_key.startswith('threshold_'):
                continue
            
            # Score based on pattern strength and statistical significance
            strength = pattern_data['pattern_strength']
            significance = pattern_data['statistical_significance']
            sample_size = pattern_data['total_matches']
            
            # Weighted score
            score = strength * significance * min(1.0, sample_size / 50)
            
            if score > best_score:
                best_score = score
                best_threshold = float(threshold_key.split('_')[1])
        
        return best_threshold
    
    def discover_market_efficiency_patterns(self, df):
        """Discover patterns related to market efficiency"""
        print("🔍 ANALYZING MARKET EFFICIENCY PATTERNS...")
        
        patterns = {}
        
        if 'bookmaker_margin' in df.columns and 'actual_winner' in df.columns:
            # Analyze different margin levels
            margin_thresholds = [0.05, 0.08, 0.10, 0.12, 0.15]
            
            for threshold in margin_thresholds:
                high_margin_mask = df['bookmaker_margin'] > threshold
                low_margin_mask = df['bookmaker_margin'] <= threshold
                
                if high_margin_mask.sum() > 10 and low_margin_mask.sum() > 10:
                    high_margin_accuracy = self.calculate_favorite_accuracy(df[high_margin_mask])
                    low_margin_accuracy = self.calculate_favorite_accuracy(df[low_margin_mask])
                    
                    patterns[f'margin_{threshold}'] = {
                        'high_margin_matches': high_margin_mask.sum(),
                        'low_margin_matches': low_margin_mask.sum(),
                        'high_margin_accuracy': high_margin_accuracy,
                        'low_margin_accuracy': low_margin_accuracy,
                        'efficiency_advantage': low_margin_accuracy - high_margin_accuracy
                    }
        
        return patterns
    
    def calculate_favorite_accuracy(self, df):
        """Calculate favorite prediction accuracy"""
        if df.empty:
            return 0.0
        
        correct = 0
        total = 0
        
        for _, match in df.iterrows():
            if all(col in match for col in ['home_win_odds', 'away_win_odds', 'draw_odds', 'actual_winner']):
                odds = [match['away_win_odds'], match['draw_odds'], match['home_win_odds']]
                favorite = np.argmin(odds)
                actual = match['actual_winner']
                
                total += 1
                if favorite == actual:
                    correct += 1
        
        return correct / total if total > 0 else 0.0
    
    def discover_temporal_patterns(self, df):
        """Discover time-based patterns"""
        print("🔍 ANALYZING TEMPORAL PATTERNS...")
        
        patterns = {}
        
        if 'start_time' in df.columns and 'actual_winner' in df.columns:
            df['start_datetime'] = pd.to_datetime(df['start_time'])
            df['hour'] = df['start_datetime'].dt.hour
            df['day_of_week'] = df['start_datetime'].dt.dayofweek
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            
            # Weekend vs weekday patterns
            weekend_matches = df[df['is_weekend'] == 1]
            weekday_matches = df[df['is_weekend'] == 0]
            
            if len(weekend_matches) > 10 and len(weekday_matches) > 10:
                weekend_accuracy = self.calculate_favorite_accuracy(weekend_matches)
                weekday_accuracy = self.calculate_favorite_accuracy(weekday_matches)
                
                patterns['weekend_vs_weekday'] = {
                    'weekend_accuracy': weekend_accuracy,
                    'weekday_accuracy': weekday_accuracy,
                    'weekend_advantage': weekend_accuracy - weekday_accuracy
                }
            
            # Time of day patterns
            time_periods = {
                'morning': (6, 12),
                'afternoon': (12, 18),
                'evening': (18, 22),
                'night': (22, 6)
            }
            
            for period_name, (start_hour, end_hour) in time_periods.items():
                if start_hour < end_hour:
                    period_mask = (df['hour'] >= start_hour) & (df['hour'] < end_hour)
                else:  # Night period crosses midnight
                    period_mask = (df['hour'] >= start_hour) | (df['hour'] < end_hour)
                
                if period_mask.sum() > 10:
                    period_accuracy = self.calculate_favorite_accuracy(df[period_mask])
                    patterns[f'{period_name}_accuracy'] = period_accuracy
        
        return patterns
    
    def discover_league_patterns(self, df):
        """Discover league-specific patterns"""
        print("🔍 ANALYZING LEAGUE-SPECIFIC PATTERNS...")
        
        patterns = {}
        
        if 'league' in df.columns and 'actual_winner' in df.columns:
            league_stats = {}
            
            for league in df['league'].unique():
                league_matches = df[df['league'] == league]
                
                if len(league_matches) > 20:  # Minimum matches for analysis
                    # Basic statistics
                    total_matches = len(league_matches)
                    home_wins = (league_matches['actual_winner'] == 2).sum()
                    away_wins = (league_matches['actual_winner'] == 0).sum()
                    draws = (league_matches['actual_winner'] == 1).sum()
                    
                    home_win_rate = home_wins / total_matches
                    away_win_rate = away_wins / total_matches
                    draw_rate = draws / total_matches
                    
                    # Goal statistics
                    if 'actual_total_goals' in league_matches.columns:
                        avg_goals = league_matches['actual_total_goals'].mean()
                        goal_variance = league_matches['actual_total_goals'].var()
                    else:
                        avg_goals = 2.5
                        goal_variance = 1.0
                    
                    # BTTS rate
                    if 'actual_btts' in league_matches.columns:
                        btts_rate = league_matches['actual_btts'].mean()
                    else:
                        btts_rate = 0.5
                    
                    # Market efficiency
                    if 'bookmaker_margin' in league_matches.columns:
                        avg_margin = league_matches['bookmaker_margin'].mean()
                    else:
                        avg_margin = 0.1
                    
                    league_stats[league] = {
                        'total_matches': total_matches,
                        'home_win_rate': home_win_rate,
                        'away_win_rate': away_win_rate,
                        'draw_rate': draw_rate,
                        'avg_goals_per_match': avg_goals,
                        'goal_variance': goal_variance,
                        'btts_rate': btts_rate,
                        'avg_bookmaker_margin': avg_margin,
                        'home_advantage': home_win_rate - away_win_rate
                    }
            
            patterns['league_statistics'] = league_stats
            
            # Find leagues with strongest patterns
            if league_stats:
                # Highest home advantage
                home_advantage_leagues = sorted(league_stats.items(), 
                                              key=lambda x: x[1]['home_advantage'], reverse=True)
                patterns['highest_home_advantage'] = home_advantage_leagues[:3]
                
                # Highest scoring leagues
                high_scoring_leagues = sorted(league_stats.items(),
                                            key=lambda x: x[1]['avg_goals_per_match'], reverse=True)
                patterns['highest_scoring'] = high_scoring_leagues[:3]
                
                # Most predictable leagues (lowest goal variance)
                predictable_leagues = sorted(league_stats.items(),
                                           key=lambda x: x[1]['goal_variance'])
                patterns['most_predictable'] = predictable_leagues[:3]
        
        return patterns
    
    def discover_team_form_patterns(self, df):
        """Discover patterns related to team form and momentum"""
        print("🔍 ANALYZING TEAM FORM PATTERNS...")
        
        patterns = {}
        
        # Analyze form-based outcomes
        form_features = [col for col in df.columns if 'form' in col or 'momentum' in col]
        
        if form_features and 'actual_winner' in df.columns:
            # High momentum vs low momentum
            for feature in form_features:
                if df[feature].dtype in ['float64', 'int64']:
                    high_threshold = df[feature].quantile(0.75)
                    low_threshold = df[feature].quantile(0.25)
                    
                    high_mask = df[feature] > high_threshold
                    low_mask = df[feature] < low_threshold
                    
                    if high_mask.sum() > 10 and low_mask.sum() > 10:
                        high_accuracy = self.calculate_favorite_accuracy(df[high_mask])
                        low_accuracy = self.calculate_favorite_accuracy(df[low_mask])
                        
                        patterns[f'{feature}_impact'] = {
                            'high_momentum_accuracy': high_accuracy,
                            'low_momentum_accuracy': low_accuracy,
                            'momentum_advantage': high_accuracy - low_accuracy
                        }
        
        return patterns
    
    def discover_clustering_patterns(self, df):
        """Discover patterns using unsupervised clustering"""
        print("🔍 DISCOVERING CLUSTERING PATTERNS...")
        
        patterns = {}
        
        # Select numerical features for clustering
        numerical_features = df.select_dtypes(include=[np.number]).columns
        feature_subset = [col for col in numerical_features if not col.startswith('actual_')]
        
        if len(feature_subset) > 5 and len(df) > 50:
            X_cluster = df[feature_subset].fillna(0)
            
            # Standardize features
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_cluster)
            
            # K-means clustering
            optimal_k = self.find_optimal_clusters(X_scaled)
            kmeans = KMeans(n_clusters=optimal_k, random_state=42)
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            # Analyze cluster characteristics
            df['cluster'] = cluster_labels
            
            cluster_analysis = {}
            for cluster_id in range(optimal_k):
                cluster_mask = df['cluster'] == cluster_id
                cluster_data = df[cluster_mask]
                
                if len(cluster_data) > 5:
                    # Cluster characteristics
                    cluster_stats = {
                        'size': len(cluster_data),
                        'avg_draw_winner_diff': cluster_data.get('draw_winner_diff', pd.Series([0])).mean(),
                        'avg_bookmaker_margin': cluster_data.get('bookmaker_margin', pd.Series([0.1])).mean(),
                    }
                    
                    # Outcome analysis
                    if 'actual_winner' in cluster_data.columns:
                        outcomes = cluster_data['actual_winner'].value_counts(normalize=True)
                        cluster_stats.update({
                            'home_win_rate': outcomes.get(2, 0),
                            'away_win_rate': outcomes.get(0, 0),
                            'draw_rate': outcomes.get(1, 0)
                        })
                        
                        # Calculate favorite accuracy for this cluster
                        cluster_stats['favorite_accuracy'] = self.calculate_favorite_accuracy(cluster_data)
                    
                    cluster_analysis[f'cluster_{cluster_id}'] = cluster_stats
            
            patterns['clustering_analysis'] = cluster_analysis
            
            # Find most profitable clusters
            if cluster_analysis:
                profitable_clusters = []
                for cluster_id, stats in cluster_analysis.items():
                    if 'favorite_accuracy' in stats and stats['size'] > 10:
                        profitable_clusters.append((cluster_id, stats['favorite_accuracy'], stats['size']))
                
                profitable_clusters.sort(key=lambda x: x[1], reverse=True)
                patterns['most_profitable_clusters'] = profitable_clusters[:3]
        
        return patterns
    
    def find_optimal_clusters(self, X, max_k=10):
        """Find optimal number of clusters using elbow method"""
        if len(X) < 20:
            return 2
        
        inertias = []
        k_range = range(2, min(max_k + 1, len(X) // 5))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
        
        # Find elbow point
        if len(inertias) > 2:
            # Calculate rate of change
            rate_of_change = np.diff(inertias)
            elbow_point = np.argmax(rate_of_change) + 2  # +2 because we start from k=2
            return min(elbow_point, max_k)
        
        return 3  # Default
    
    def build_advanced_decision_trees(self, df):
        """Build decision trees for pattern discovery"""
        print("🌳 BUILDING ADVANCED DECISION TREES...")
        
        if 'actual_winner' in df.columns:
            # Features for decision tree
            tree_features = []
            
            # Odds-based features
            if all(col in df.columns for col in ['draw_winner_diff', 'bookmaker_margin', 'odds_entropy']):
                tree_features.extend(['draw_winner_diff', 'bookmaker_margin', 'odds_entropy'])
            
            # Performance features
            performance_features = [col for col in df.columns if 'performance' in col or 'rating' in col]
            tree_features.extend(performance_features[:5])  # Top 5 performance features
            
            # Market features
            market_features = [col for col in df.columns if 'market' in col or 'efficiency' in col]
            tree_features.extend(market_features[:3])
            
            # Remove duplicates and ensure features exist
            tree_features = list(set([f for f in tree_features if f in df.columns]))
            
            if len(tree_features) > 3:
                X_tree = df[tree_features].fillna(0)
                y_tree = df['actual_winner']
                
                # Build interpretable decision tree
                tree = DecisionTreeClassifier(
                    max_depth=6,
                    min_samples_split=20,
                    min_samples_leaf=10,
                    random_state=42
                )
                
                tree.fit(X_tree, y_tree)
                
                # Extract rules
                rules = self.extract_decision_rules(tree, tree_features)
                
                return {
                    'tree_model': tree,
                    'features_used': tree_features,
                    'decision_rules': rules,
                    'tree_accuracy': tree.score(X_tree, y_tree)
                }
        
        return {}
    
    def extract_decision_rules(self, tree, feature_names):
        """Extract human-readable decision rules from tree"""
        rules = []
        
        def recurse(node, depth, condition):
            if tree.tree_.children_left[node] == tree.tree_.children_right[node]:
                # Leaf node
                values = tree.tree_.value[node][0]
                predicted_class = np.argmax(values)
                confidence = values[predicted_class] / np.sum(values)
                
                class_names = ['Away Win', 'Draw', 'Home Win']
                
                rule = {
                    'condition': condition,
                    'prediction': class_names[predicted_class],
                    'confidence': confidence,
                    'samples': tree.tree_.n_node_samples[node]
                }
                rules.append(rule)
            else:
                # Internal node
                feature = feature_names[tree.tree_.feature[node]]
                threshold = tree.tree_.threshold[node]
                
                # Left child (<=)
                left_condition = condition + f" AND {feature} <= {threshold:.3f}" if condition else f"{feature} <= {threshold:.3f}"
                recurse(tree.tree_.children_left[node], depth + 1, left_condition)
                
                # Right child (>)
                right_condition = condition + f" AND {feature} > {threshold:.3f}" if condition else f"{feature} > {threshold:.3f}"
                recurse(tree.tree_.children_right[node], depth + 1, right_condition)
        
        recurse(0, 0, "")
        
        # Sort by confidence and sample size
        rules.sort(key=lambda x: (x['confidence'], x['samples']), reverse=True)
        
        return rules[:10]  # Top 10 rules
    
    def discover_correlation_patterns(self, df):
        """Discover correlation patterns between features and outcomes"""
        print("🔍 ANALYZING CORRELATION PATTERNS...")
        
        patterns = {}
        
        if 'actual_winner' in df.columns:
            # Calculate correlations with outcome
            numerical_features = df.select_dtypes(include=[np.number]).columns
            feature_subset = [col for col in numerical_features if not col.startswith('actual_') or col == 'actual_winner']
            
            if len(feature_subset) > 5:
                correlation_matrix = df[feature_subset].corr()
                
                # Find strongest correlations with actual_winner
                winner_correlations = correlation_matrix['actual_winner'].abs().sort_values(ascending=False)
                
                # Exclude self-correlation
                winner_correlations = winner_correlations[winner_correlations.index != 'actual_winner']
                
                patterns['strongest_predictors'] = {
                    'features': winner_correlations.head(10).to_dict(),
                    'top_positive': winner_correlations.head(5).to_dict(),
                    'top_negative': winner_correlations.tail(5).to_dict()
                }
                
                # Find feature clusters (highly correlated features)
                feature_correlations = correlation_matrix.drop('actual_winner', axis=0).drop('actual_winner', axis=1)
                high_corr_pairs = []
                
                for i in range(len(feature_correlations.columns)):
                    for j in range(i+1, len(feature_correlations.columns)):
                        corr_value = feature_correlations.iloc[i, j]
                        if abs(corr_value) > 0.8:
                            high_corr_pairs.append((
                                feature_correlations.columns[i],
                                feature_correlations.columns[j],
                                corr_value
                            ))
                
                patterns['high_correlation_pairs'] = high_corr_pairs[:10]
        
        return patterns
    
    def discover_anomaly_patterns(self, df):
        """Discover anomalous patterns that might indicate value bets"""
        print("🔍 ANALYZING ANOMALY PATTERNS...")
        
        patterns = {}
        
        # Odds anomalies
        if all(col in df.columns for col in ['home_win_odds', 'away_win_odds', 'draw_odds']):
            # Find matches with unusual odds distributions
            df['odds_std'] = df[['home_win_odds', 'away_win_odds', 'draw_odds']].std(axis=1)
            df['odds_range'] = df[['home_win_odds', 'away_win_odds', 'draw_odds']].max(axis=1) - \
                              df[['home_win_odds', 'away_win_odds', 'draw_odds']].min(axis=1)
            
            # High variance odds (unusual distributions)
            high_variance_threshold = df['odds_std'].quantile(0.9)
            high_variance_matches = df[df['odds_std'] > high_variance_threshold]
            
            if len(high_variance_matches) > 5 and 'actual_winner' in df.columns:
                anomaly_accuracy = self.calculate_favorite_accuracy(high_variance_matches)
                normal_accuracy = self.calculate_favorite_accuracy(df[df['odds_std'] <= high_variance_threshold])
                
                patterns['odds_anomalies'] = {
                    'anomaly_matches': len(high_variance_matches),
                    'anomaly_accuracy': anomaly_accuracy,
                    'normal_accuracy': normal_accuracy,
                    'anomaly_advantage': anomaly_accuracy - normal_accuracy
                }
        
        # Performance anomalies
        performance_features = [col for col in df.columns if 'performance' in col and 'diff' in col]
        
        for feature in performance_features:
            if df[feature].dtype in ['float64', 'int64']:
                # Find extreme values
                extreme_threshold = df[feature].quantile(0.95)
                extreme_mask = abs(df[feature]) > extreme_threshold
                
                if extreme_mask.sum() > 5 and 'actual_winner' in df.columns:
                    extreme_accuracy = self.calculate_favorite_accuracy(df[extreme_mask])
                    normal_accuracy = self.calculate_favorite_accuracy(df[~extreme_mask])
                    
                    patterns[f'{feature}_extremes'] = {
                        'extreme_matches': extreme_mask.sum(),
                        'extreme_accuracy': extreme_accuracy,
                        'normal_accuracy': normal_accuracy,
                        'extreme_advantage': extreme_accuracy - normal_accuracy
                    }
        
        return patterns
    
    def generate_pattern_report(self, all_patterns):
        """Generate comprehensive pattern analysis report"""
        print(f"\n{'='*70}")
        print(f"🧠 ADVANCED PATTERN ANALYSIS REPORT")
        print(f"{'='*70}")
        
        # 1. Draw-difference patterns (your specific pattern)
        if 'draw_difference' in all_patterns:
            draw_patterns = all_patterns['draw_difference']
            print(f"\n🎯 DRAW-DIFFERENCE PATTERN ANALYSIS:")
            
            if 'optimal_threshold' in draw_patterns:
                optimal = draw_patterns['optimal_threshold']
                print(f"   Optimal Threshold: {optimal}")
                
                if f'threshold_{optimal}' in draw_patterns:
                    pattern_data = draw_patterns[f'threshold_{optimal}']
                    print(f"   Matches: {pattern_data['total_matches']}")
                    print(f"   Non-draw Rate: {pattern_data['non_draw_rate']:.1%}")
                    print(f"   Pattern Strength: {pattern_data['pattern_strength']:+.3f}")
                    print(f"   Statistical Significance: {pattern_data['statistical_significance']:.3f}")
                    
                    if pattern_data['pattern_strength'] > 0.1:
                        print(f"   ✅ STRONG PATTERN DETECTED - Use for betting strategy")
                    elif pattern_data['pattern_strength'] > 0.05:
                        print(f"   ⚠️  MODERATE PATTERN - Use with caution")
                    else:
                        print(f"   ❌ WEAK PATTERN - Not reliable for betting")
        
        # 2. Market efficiency patterns
        if 'market_efficiency' in all_patterns:
            print(f"\n💰 MARKET EFFICIENCY PATTERNS:")
            for pattern_name, pattern_data in all_patterns['market_efficiency'].items():
                if isinstance(pattern_data, dict) and 'efficiency_advantage' in pattern_data:
                    print(f"   {pattern_name}: {pattern_data['efficiency_advantage']:+.3f} advantage")
        
        # 3. Temporal patterns
        if 'temporal' in all_patterns:
            print(f"\n⏰ TEMPORAL PATTERNS:")
            temporal_patterns = all_patterns['temporal']
            
            if 'weekend_vs_weekday' in temporal_patterns:
                weekend_data = temporal_patterns['weekend_vs_weekday']
                print(f"   Weekend Advantage: {weekend_data['weekend_advantage']:+.3f}")
            
            # Time of day patterns
            time_periods = ['morning', 'afternoon', 'evening', 'night']
            for period in time_periods:
                accuracy_key = f'{period}_accuracy'
                if accuracy_key in temporal_patterns:
                    print(f"   {period.title()} Accuracy: {temporal_patterns[accuracy_key]:.1%}")
        
        # 4. League patterns
        if 'league' in all_patterns and 'league_statistics' in all_patterns['league']:
            print(f"\n🏆 LEAGUE PATTERNS:")
            
            # Highest home advantage
            if 'highest_home_advantage' in all_patterns['league']:
                print(f"   Strongest Home Advantage:")
                for league, stats in all_patterns['league']['highest_home_advantage']:
                    print(f"     {league}: {stats['home_advantage']:+.3f}")
            
            # Highest scoring
            if 'highest_scoring' in all_patterns['league']:
                print(f"   Highest Scoring Leagues:")
                for league, stats in all_patterns['league']['highest_scoring']:
                    print(f"     {league}: {stats['avg_goals_per_match']:.2f} goals/match")
        
        # 5. Clustering insights
        if 'clustering' in all_patterns and 'most_profitable_clusters' in all_patterns['clustering']:
            print(f"\n🎲 CLUSTERING INSIGHTS:")
            print(f"   Most Profitable Clusters:")
            for cluster_id, accuracy, size in all_patterns['clustering']['most_profitable_clusters']:
                print(f"     {cluster_id}: {accuracy:.1%} accuracy ({size} matches)")
        
        # 6. Anomaly patterns
        if 'anomalies' in all_patterns:
            print(f"\n🚨 ANOMALY PATTERNS:")
            anomaly_patterns = all_patterns['anomalies']
            
            for pattern_name, pattern_data in anomaly_patterns.items():
                if isinstance(pattern_data, dict) and 'anomaly_advantage' in pattern_data:
                    advantage = pattern_data['anomaly_advantage']
                    if abs(advantage) > 0.05:
                        print(f"   {pattern_name}: {advantage:+.3f} advantage")
        
        return all_patterns
    
    def create_betting_strategy(self, patterns):
        """Create betting strategy based on discovered patterns"""
        strategy = {
            'rules': [],
            'confidence_modifiers': {},
            'avoid_conditions': [],
            'value_bet_indicators': []
        }
        
        # Rule 1: Your draw-difference pattern
        if 'draw_difference' in patterns and 'optimal_threshold' in patterns['draw_difference']:
            threshold = patterns['draw_difference']['optimal_threshold']
            pattern_data = patterns['draw_difference'].get(f'threshold_{threshold}', {})
            
            if pattern_data.get('pattern_strength', 0) > 0.1:
                strategy['rules'].append({
                    'name': 'High Draw Difference Rule',
                    'condition': f'draw_winner_diff > {threshold}',
                    'action': 'Bet against draw',
                    'confidence_boost': 0.15,
                    'strength': pattern_data['pattern_strength']
                })
        
        # Rule 2: Market efficiency
        if 'market_efficiency' in patterns:
            for pattern_name, pattern_data in patterns['market_efficiency'].items():
                if isinstance(pattern_data, dict) and pattern_data.get('efficiency_advantage', 0) > 0.1:
                    strategy['rules'].append({
                        'name': f'Market Efficiency Rule ({pattern_name})',
                        'condition': pattern_name,
                        'action': 'Increase confidence in low-margin markets',
                        'confidence_boost': 0.1
                    })
        
        # Rule 3: Clustering-based rules
        if 'clustering' in patterns and 'most_profitable_clusters' in patterns['clustering']:
            for cluster_id, accuracy, size in patterns['clustering']['most_profitable_clusters']:
                if accuracy > 0.7 and size > 20:
                    strategy['rules'].append({
                        'name': f'High-Accuracy Cluster Rule',
                        'condition': f'Match belongs to {cluster_id}',
                        'action': 'Increase confidence',
                        'confidence_boost': 0.1,
                        'accuracy': accuracy
                    })
        
        # Avoid conditions
        if 'anomalies' in patterns:
            for pattern_name, pattern_data in patterns['anomalies'].items():
                if isinstance(pattern_data, dict) and pattern_data.get('anomaly_advantage', 0) < -0.1:
                    strategy['avoid_conditions'].append({
                        'condition': pattern_name,
                        'reason': 'Negative anomaly advantage',
                        'confidence_penalty': 0.15
                    })
        
        return strategy
    
    def apply_pattern_strategy(self, match_features, base_predictions, base_confidences, strategy):
        """Apply discovered patterns to adjust predictions"""
        adjusted_confidences = base_confidences.copy()
        strategy_applied = []
        
        # Apply rules
        for rule in strategy['rules']:
            rule_triggered = self.check_rule_condition(match_features, rule['condition'])
            
            if rule_triggered:
                # Apply confidence boost
                boost = rule.get('confidence_boost', 0)
                for target in adjusted_confidences:
                    adjusted_confidences[target] = min(0.95, adjusted_confidences[target] + boost)
                
                strategy_applied.append(rule['name'])
        
        # Apply avoid conditions
        for avoid_condition in strategy['avoid_conditions']:
            condition_met = self.check_rule_condition(match_features, avoid_condition['condition'])
            
            if condition_met:
                # Apply confidence penalty
                penalty = avoid_condition.get('confidence_penalty', 0)
                for target in adjusted_confidences:
                    adjusted_confidences[target] = max(0.05, adjusted_confidences[target] - penalty)
                
                strategy_applied.append(f"AVOID: {avoid_condition['condition']}")
        
        return adjusted_confidences, strategy_applied
    
    def check_rule_condition(self, match_features, condition):
        """Check if a rule condition is met"""
        try:
            # Simple condition checking
            if 'draw_winner_diff >' in condition:
                threshold = float(condition.split('>')[-1].strip())
                return match_features.get('draw_winner_diff', 0) > threshold
            
            elif 'bookmaker_margin <' in condition:
                threshold = float(condition.split('<')[-1].strip())
                return match_features.get('bookmaker_margin', 0.1) < threshold
            
            elif 'market_efficiency >' in condition:
                threshold = float(condition.split('>')[-1].strip())
                return match_features.get('market_efficiency', 0.5) > threshold
            
            # Add more condition types as needed
            
        except:
            pass
        
        return False

def run_comprehensive_pattern_analysis(predictor):
    """Run comprehensive pattern analysis on all data"""
    if predictor.base_predictor.matched_data.empty:
        print("No matched data available for pattern analysis")
        return
    
    analyzer = AdvancedPatternAnalyzer()
    df = predictor.base_predictor.matched_data.copy()
    
    print("🔬 RUNNING COMPREHENSIVE PATTERN ANALYSIS...")
    
    # Discover all pattern types
    all_patterns = {}
    
    # 1. Your specific draw-difference patterns
    all_patterns['draw_difference'] = analyzer.discover_draw_difference_patterns(df)
    
    # 2. Market efficiency patterns
    all_patterns['market_efficiency'] = analyzer.discover_market_efficiency_patterns(df)
    
    # 3. Temporal patterns
    all_patterns['temporal'] = analyzer.discover_temporal_patterns(df)
    
    # 4. League patterns
    all_patterns['league'] = analyzer.discover_league_patterns(df)
    
    # 5. Team form patterns
    all_patterns['team_form'] = analyzer.discover_team_form_patterns(df)
    
    # 6. Clustering patterns
    all_patterns['clustering'] = analyzer.discover_clustering_patterns(df)
    
    # 7. Correlation patterns
    all_patterns['correlations'] = analyzer.discover_correlation_patterns(df)
    
    # 8. Anomaly patterns
    all_patterns['anomalies'] = analyzer.discover_anomaly_patterns(df)
    
    # 9. Decision tree patterns
    all_patterns['decision_trees'] = analyzer.build_advanced_decision_trees(df)
    
    # Generate comprehensive report
    analyzer.generate_pattern_report(all_patterns)
    
    # Create betting strategy
    strategy = analyzer.create_betting_strategy(all_patterns)
    
    print(f"\n🎯 GENERATED BETTING STRATEGY:")
    print(f"   Rules: {len(strategy['rules'])}")
    print(f"   Avoid Conditions: {len(strategy['avoid_conditions'])}")
    
    for i, rule in enumerate(strategy['rules'], 1):
        print(f"   {i}. {rule['name']}: {rule['action']}")
    
    return all_patterns, strategy

if __name__ == "__main__":
    # Example usage
    predictor = UltraAdvancedSportsPredictor()
    
    if not predictor.load_ultra_models():
        predictor.train_ultra_advanced_models()
        predictor.save_ultra_models()
    
    # Run pattern analysis
    patterns, strategy = run_comprehensive_pattern_analysis(predictor)
    
    print("\n🚀 Ultra-Advanced Sports Prediction System Ready!")
    print("This system includes:")
    print("- Your specific draw-difference pattern with optimal threshold detection")
    print("- 10+ advanced ML algorithms in ensemble")
    print("- Deep neural networks with attention mechanisms")
    print("- Reinforcement learning for continuous improvement")
    print("- Meta-learning for prediction quality assessment")
    print("- Comprehensive pattern discovery and analysis")
    print("- Advanced feature engineering with 100+ features")
    print("- Hyperparameter optimization")
    print("- Anomaly detection for value bet identification")