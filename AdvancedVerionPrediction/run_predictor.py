#!/usr/bin/env python3
"""
Ultra-Advanced Sports Prediction System
Main execution script with enhanced interface
"""

import sys
import os
from advanced_predictor import UltraAdvancedSportsPredictor
from pattern_analyzer import run_comprehensive_pattern_analysis
import warnings
warnings.filterwarnings('ignore')

def main():
    """Main execution with enhanced interface"""
    print("🚀 INITIALIZING ULTRA-ADVANCED SPORTS PREDICTION SYSTEM...")
    
    # Initialize predictor
    predictor = UltraAdvancedSportsPredictor()
    
    # Load or train models
    print("Loading models...")
    if not predictor.load_ultra_models():
        print("No existing models found. Training new ultra-advanced models...")
        print("This may take several minutes...")
        predictor.train_ultra_advanced_models()
        predictor.save_ultra_models()
        print("✅ Training completed and models saved!")
    else:
        print("✅ Models loaded successfully!")
    
    # Run initial pattern analysis
    print("\n🔬 Running comprehensive pattern analysis...")
    try:
        patterns, strategy = run_comprehensive_pattern_analysis(predictor)
        print("✅ Pattern analysis completed!")
    except Exception as e:
        print(f"⚠️  Pattern analysis failed: {e}")
        patterns, strategy = {}, {}
    
    # Main interface
    print(f"\n{'='*80}")
    print(f"🎯 ULTRA-ADVANCED SPORTS PREDICTION SYSTEM - READY")
    print(f"{'='*80}")
    
    print("\n🎲 AVAILABLE COMMANDS:")
    print("━" * 50)
    print("1. predict <home_team> <away_team> <league> [upcoming_file.json]")
    print("   Example: predict Arsenal Chelsea 'Premier League' upcoming.json")
    print("   Example: predict Barcelona Madrid 'La Liga'")
    print("")
    print("2. predict_all <upcoming_file.json>")
    print("   Example: predict_all today_matches.json")
    print("")
    print("3. analyze_patterns")
    print("   Run comprehensive pattern analysis")
    print("")
    print("4. team_analysis <team_name>")
    print("   Example: team_analysis Arsenal")
    print("")
    print("5. retrain")
    print("   Retrain all models with latest data")
    print("")
    print("6. performance")
    print("   Show detailed performance metrics")
    print("")
    print("7. save")
    print("   Save current models and data")
    print("")
    print("8. help")
    print("   Show this help message")
    print("")
    print("9. quit")
    print("   Exit the system")
    
    print(f"\n🧠 SYSTEM FEATURES:")
    print("✅ Advanced ensemble learning (10+ algorithms)")
    print("✅ Deep neural networks with attention")
    print("✅ Your specific draw-difference pattern detection")
    print("✅ Reinforcement learning for continuous improvement")
    print("✅ Meta-learning for prediction quality assessment")
    print("✅ Advanced pattern recognition")
    print("✅ Market efficiency analysis")
    print("✅ Hyperparameter optimization")
    print("✅ Anomaly detection for value bets")
    
    # Interactive loop
    while True:
        try:
            print(f"\n{'─'*50}")
            command = input("🎯 Enter command: ").strip()
            
            if not command:
                continue
            
            parts = command.split()
            cmd = parts[0].lower()
            
            if cmd == 'quit' or cmd == 'exit':
                print("👋 Goodbye! Thanks for using the Ultra-Advanced Prediction System!")
                break
                
            elif cmd == 'help':
                print("\n📖 COMMAND HELP:")
                print("predict - Make advanced predictions for specific matches")
                print("predict_all - Predict all matches in an upcoming file")
                print("analyze_patterns - Discover hidden patterns in data")
                print("team_analysis - Detailed analysis of specific team")
                print("retrain - Update models with latest data")
                print("performance - Show model accuracy and performance")
                print("save - Save current state")
                
            elif cmd == 'predict':
                if len(parts) >= 4:
                    home_team = parts[1]
                    away_team = parts[2]
                    league = ' '.join(parts[3:-1]) if len(parts) > 4 and parts[-1].endswith('.json') else ' '.join(parts[3:])
                    upcoming_file = parts[-1] if len(parts) > 4 and parts[-1].endswith('.json') else None
                    
                    print(f"\n🔮 PREDICTING: {home_team} vs {away_team} ({league})")
                    if upcoming_file:
                        print(f"📊 Using current odds from: {upcoming_file}")
                    
                    result = predictor.ultra_predict(home_team, away_team, league, upcoming_file)
                    
                    if result:
                        print("\n✅ Prediction completed!")
                    else:
                        print("\n❌ Prediction failed - check team names and data availability")
                else:
                    print("❌ Usage: predict <home_team> <away_team> <league> [upcoming_file.json]")
                    print("   Example: predict Arsenal Chelsea 'Premier League' upcoming.json")
                    
            elif cmd == 'predict_all':
                if len(parts) >= 2:
                    upcoming_file = parts[1]
                    print(f"\n🔮 PREDICTING ALL MATCHES FROM: {upcoming_file}")
                    
                    try:
                        # Use base predictor method for all matches
                        predictions = predictor.base_predictor.predict_all_upcoming_matches(upcoming_file)
                        if predictions:
                            predictor.base_predictor.display_upcoming_predictions_summary(predictions)
                            print("\n✅ All predictions completed!")
                        else:
                            print("❌ No matches found or prediction failed")
                    except Exception as e:
                        print(f"❌ Error predicting all matches: {e}")
                else:
                    print("❌ Usage: predict_all <upcoming_file.json>")
                    
            elif cmd == 'analyze_patterns':
                print("\n🔬 RUNNING COMPREHENSIVE PATTERN ANALYSIS...")
                try:
                    patterns, strategy = run_comprehensive_pattern_analysis(predictor)
                    print("\n✅ Pattern analysis completed!")
                except Exception as e:
                    print(f"❌ Pattern analysis failed: {e}")
                    
            elif cmd == 'team_analysis':
                if len(parts) >= 2:
                    team_name = ' '.join(parts[1:])
                    print(f"\n📊 ANALYZING TEAM: {team_name}")
                    
                    if team_name in predictor.base_predictor.team_stats:
                        stats = predictor.base_predictor.team_stats[team_name]
                        
                        print(f"\n🏆 {team_name.upper()} - COMPREHENSIVE ANALYSIS:")
                        print(f"   Performance Rating: {stats.get('performance_rating', 0):.3f}")
                        print(f"   Total Matches: {stats.get('total_matches', 0)}")
                        print(f"   Win Rate: {stats.get('win_rate', 0):.1%}")
                        print(f"   Home Win Rate: {stats.get('home_win_rate', 0):.1%}")
                        print(f"   Away Win Rate: {stats.get('away_win_rate', 0):.1%}")
                        print(f"   Average Goals Scored: {stats.get('avg_goals_scored', 0):.2f}")
                        print(f"   Average Goals Conceded: {stats.get('avg_goals_conceded', 0):.2f}")
                        print(f"   Goal Difference: {stats.get('goal_difference', 0):+.2f}")
                        print(f"   Form Momentum: {stats.get('form_momentum', 0):.3f}")
                        print(f"   Consistency Score: {stats.get('consistency_score', 0):.3f}")
                        print(f"   Current Win Streak: {stats.get('current_win_streak', 0)}")
                        print(f"   Current Loss Streak: {stats.get('current_loss_streak', 0)}")
                        
                        if stats.get('leagues'):
                            print(f"   Leagues: {', '.join(stats['leagues'])}")
                    else:
                        print(f"❌ No data found for team: {team_name}")
                        print("Available teams:")
                        available_teams = list(predictor.base_predictor.team_stats.keys())[:10]
                        for team in available_teams:
                            print(f"   - {team}")
                        if len(predictor.base_predictor.team_stats) > 10:
                            print(f"   ... and {len(predictor.base_predictor.team_stats) - 10} more")
                else:
                    print("❌ Usage: team_analysis <team_name>")
                    
            elif cmd == 'retrain':
                print("\n🔄 RETRAINING ULTRA-ADVANCED MODELS...")
                print("This may take several minutes...")
                try:
                    predictor.train_ultra_advanced_models()
                    predictor.save_ultra_models()
                    print("✅ Retraining completed!")
                except Exception as e:
                    print(f"❌ Retraining failed: {e}")
                    
            elif cmd == 'performance':
                print("\n📈 MODEL PERFORMANCE REPORT:")
                try:
                    report = predictor.base_predictor.get_model_performance_report()
                    print(report)
                except Exception as e:
                    print(f"❌ Error generating performance report: {e}")
                    
            elif cmd == 'save':
                print("\n💾 SAVING MODELS AND DATA...")
                try:
                    predictor.save_ultra_models()
                    print("✅ Models saved successfully!")
                except Exception as e:
                    print(f"❌ Save failed: {e}")
                    
            else:
                print(f"❌ Unknown command: {cmd}")
                print("💡 Type 'help' for available commands")
                
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            print("💡 Please try again or type 'help' for available commands")

if __name__ == "__main__":
    main()