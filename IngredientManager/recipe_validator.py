import pandas as pd
import numpy as np
import json
import os
import time
import ast
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Import from the same directory
from recipe_recommender import RecipeRecommender


class RecipeValidator:
    def __init__(self, ingredients_file=None, inappropriate_terms_file=None):
        """
        Initialize the recipe validator with ingredient lists and filters.

        Args:
            ingredients_file: Path to a JSON file with ingredient lists for testing
            inappropriate_terms_file: Path to a text file with inappropriate terms to filter
        """
        self.recommender = None

        # Load test ingredients
        self.test_ingredients = self._load_ingredients(ingredients_file)

        # Load inappropriate terms list
        self.inappropriate_terms = self._load_inappropriate_terms(inappropriate_terms_file)

        # For storing validation results
        self.validation_results = []

    def _load_ingredients(self, file_path):
        """Load ingredient test sets from a JSON file or create default ones."""
        if file_path and os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    ingredients = json.load(f)
                print(f"Loaded {len(ingredients)} ingredient test sets from {file_path}")
                return ingredients
            except Exception as e:
                print(f"Error loading ingredients file: {e}")
        else:
            print(f"Test Ingredient file not found: {file_path}")
        return {}

    def _load_inappropriate_terms(self, file_path):
        """Load inappropriate terms from a file or create default ones."""
        if file_path and os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    terms = [line.strip().lower() for line in f if line.strip()]
                print(f"Loaded {len(terms)} inappropriate terms from {file_path}")
                return terms
            except Exception as e:
                print(f"Error loading inappropriate terms file: {e}")

        else:
            print(f"Inappropriate Terms file not found: {file_path}")
        return {}

    def load_recommender(self, index_path, df_path):
        """Load an existing FAISS index and DataFrame."""
        self.recommender = RecipeRecommender()
        return self.recommender.load_model(index_path, df_path)

    def check_ingredient_coverage(self):
        """Check how many of our test ingredients exist in the dataset."""
        if not self.recommender or self.recommender.df is None:
            raise ValueError("Recommender not loaded. Call load_recommender first.")

        # Extract all unique ingredients from the dataset
        all_dataset_ingredients = set()
        for ner_list in self.recommender.df['NER']:
            try:
                ingredients = ast.literal_eval(ner_list) if isinstance(ner_list, str) else []
                all_dataset_ingredients.update([ing.lower() for ing in ingredients])
            except:
                continue

        # Check coverage for each test set
        coverage_results = {}
        for name, ingredients in self.test_ingredients.items():
            found = [ing for ing in ingredients if ing.lower() in all_dataset_ingredients]
            found_count = len(found)
            total_count = len(ingredients)
            coverage_pct = 0
            if total_count > 0:
                coverage_pct = round(found_count / total_count * 100, 2)

            coverage_results[name] = {
                'total': total_count,
                'found': found_count,
                'coverage_pct': coverage_pct,
                'missing': [ing for ing in ingredients if ing.lower() not in all_dataset_ingredients]
            }

        return coverage_results

    def check_for_inappropriate_content(self, recipe):
        """
        Check if a recipe contains any inappropriate content.

        Args:
            recipe: Dictionary with recipe info including 'title' and 'ingredients'

        Returns:
            Tuple of (is_flagged, [matched_terms])
        """
        # Get text to check (title and ingredients joined)
        recipe_text = recipe['title'].lower() + ' ' + ' '.join(recipe['ingredients']).lower()

        # Check for inappropriate terms
        matched_terms = [term for term in self.inappropriate_terms
                         if term.lower() in recipe_text]

        return (len(matched_terms) > 0, matched_terms)

    def validate_recommendation_relevance(self, ingredients, recipe_ingredients):
        """
        Validate that the recommended recipe is relevant to the input ingredients.

        Args:
            ingredients: List of input ingredients
            recipe_ingredients: List of ingredients in the recommended recipe

        Returns:
            Dict with validation metrics
        """
        # Convert all to lowercase for comparison
        query_ingredients = [ing.lower() for ing in ingredients]
        recipe_ingredients_lower = [ing.lower() for ing in recipe_ingredients]

        # Calculate overlap stats
        common_ingredients = set(query_ingredients) & set(recipe_ingredients_lower)
        common_count = len(common_ingredients)

        # Calculate percentages with safety checks
        ingredients_matched_pct = 0
        recipe_coverage_pct = 0

        if query_ingredients:
            ingredients_matched_pct = round(common_count / len(query_ingredients) * 100, 2)

        if recipe_ingredients_lower:
            recipe_coverage_pct = round(common_count / len(recipe_ingredients_lower) * 100, 2)

        return {
            'ingredients_matched': common_count,
            'ingredients_matched_pct': ingredients_matched_pct,
            'recipe_coverage_pct': recipe_coverage_pct,
            'common_ingredients': list(common_ingredients)
        }

    def validate_recommendation_coherence(self, recommendations):
        """
        Validate the coherence of recommendations (do they make sense together).

        Args:
            recommendations: List of recipe dictionaries

        Returns:
            Coherence score (0-1) and analysis
        """
        if not recommendations:
            return {'coherence_score': 0, 'analysis': "No recommendations to analyze"}

        # Extract recipe titles and create document vectors
        titles = [recipe['title'] for recipe in recommendations]

        if len(titles) < 2:
            return {'coherence_score': 1.0, 'analysis': "Only one recipe, coherence perfect by default"}

        # Use CountVectorizer to create document vectors
        vectorizer = CountVectorizer(stop_words='english')

        try:
            title_vectors = vectorizer.fit_transform(titles)
            # Compute pairwise similarities
            similarities = cosine_similarity(title_vectors)

            # Get average similarity (excluding self-similarity on diagonal)
            n = similarities.shape[0]
            total_sim = 0
            if n > 1:  # Avoid division by zero
                total_sim = (similarities.sum() - n) / (n * (n - 1))

            # Analyze feature similarity
            analysis = ""
            if total_sim < 0.1:
                analysis = "Very low coherence - recipes seem unrelated"
            elif total_sim < 0.3:
                analysis = "Low coherence - recipes somewhat related"
            elif total_sim < 0.6:
                analysis = "Moderate coherence - recipes show clear pattern"
            else:
                analysis = "High coherence - recipes strongly related"

            return {
                'coherence_score': float(total_sim),
                'analysis': analysis
            }
        except Exception as e:
            return {'coherence_score': 0, 'analysis': f"Could not compute coherence: {str(e)}"}

    def validate_single_test_case(self, test_name, ingredients, k=10):
        """
        Validate recommendations for a single test case.

        Args:
            test_name: Name of the test case
            ingredients: List of ingredients to test
            k: Number of recommendations to request

        Returns:
            Dictionary with validation results
        """
        if not self.recommender:
            raise ValueError("Recommender not loaded")

        print(f"Testing: {test_name} - {ingredients}")

        # Get recommendations
        start_time = time.time()
        results = self.recommender.recommend(ingredients, k=k)
        query_time = time.time() - start_time

        # Validate each recommendation
        validated_results = []
        flagged_count = 0
        relevance_scores = []

        for recipe in results:
            # Check for inappropriate content
            is_flagged, matched_terms = self.check_for_inappropriate_content(recipe)
            if is_flagged:
                flagged_count += 1

            # Validate relevance
            relevance = self.validate_recommendation_relevance(ingredients, recipe['ingredients'])
            relevance_scores.append(relevance['ingredients_matched_pct'])

            validated_results.append({
                'recipe': recipe,
                'is_flagged': is_flagged,
                'matched_terms': matched_terms,
                'relevance': relevance
            })

        # Validate coherence across recommendations
        coherence = self.validate_recommendation_coherence(results)

        # Calculate average relevance score safely
        avg_relevance_score = 0
        if relevance_scores:
            avg_relevance_score = sum(relevance_scores) / len(relevance_scores)

        # Prepare validation summary
        validation_result = {
            'test_name': test_name,
            'ingredients': ingredients,
            'query_time': query_time,
            'results_count': len(results),
            'flagged_count': flagged_count,
            'avg_relevance_score': avg_relevance_score,
            'coherence': coherence,
            'validated_results': validated_results
        }

        # Apply coherence threshold filter
        coherence_threshold = 0.5  # adjust as needed
        if coherence['coherence_score'] < coherence_threshold:
            validation_result['filtered_out'] = True
            validation_result['validated_results'] = []
            validation_result['results_count'] = 0
            validation_result['flagged_count'] = 0
            validation_result['avg_relevance_score'] = 0

        self.validation_results.append(validation_result)

        return validation_result

    def run_validation(self, test_selection=None, k=10):
        """
        Run validation on multiple test cases.

        Args:
            test_selection: List of test names to run, or None for all
            k: Number of recommendations to request

        Returns:
            DataFrame with validation results summary
        """
        if k <= 0:
            print("Skipping validation (k=0)")
            return pd.DataFrame()

        if not self.recommender:
            raise ValueError("Recommender not loaded")

        # Select which tests to run
        if test_selection:
            tests = {name: self.test_ingredients[name] for name in test_selection if name in self.test_ingredients}
        else:
            tests = self.test_ingredients

        if not tests:
            print("No valid tests to run")
            return pd.DataFrame()

        self.validation_results = []
        print(f"Running {len(tests)} validation tests...")

        for name, ingredients in tests.items():
            result = self.validate_single_test_case(name, ingredients, k=k)
            print(f"  {name}: {result['results_count']} results, {result['flagged_count']} flagged, " +
                  f"avg relevance: {result['avg_relevance_score']:.2f}%, coherence: {result['coherence']['coherence_score']:.2f}")

        # Create and return summary DataFrame
        return self.create_summary_dataframe()

    def create_summary_dataframe(self):
        """
        Create a summary DataFrame from validation results.

        Returns:
            DataFrame with validation summary
        """
        if not self.validation_results:
            return pd.DataFrame()

        summary = []
        for result in self.validation_results:
            # Calculate flagged percentage safely
            flagged_pct = "N/A"
            if result['results_count'] > 0:
                flagged_pct = f"{(result['flagged_count'] / result['results_count'] * 100):.1f}%"

            summary.append({
                'test_name': result['test_name'],
                'ingredients': ', '.join(result['ingredients']),
                'results_count': result['results_count'],
                'query_time': f"{result['query_time']:.4f}s",
                'flagged_count': result['flagged_count'],
                'flagged_pct': flagged_pct,
                'avg_relevance': f"{result['avg_relevance_score']:.1f}%",
                'coherence_score': f"{result['coherence']['coherence_score']:.2f}",
                'coherence_analysis': result['coherence']['analysis']
            })

        return pd.DataFrame(summary)

    def save_validation_results(self, output_dir="validation_results"):
        """
        Save detailed validation results to files.

        Args:
            output_dir: Directory to save results
        """
        if not self.validation_results:
            print("No validation results to save")
            return

        os.makedirs(output_dir, exist_ok=True)

        # Save detailed results as JSON
        detailed_path = os.path.join(output_dir, "detailed_validation_results.json")
        with open(detailed_path, 'w') as f:
            # Convert numpy values to Python native types for JSON serialization
            results_copy = []
            for result in self.validation_results:
                result_copy = result.copy()
                if 'avg_relevance_score' in result_copy and isinstance(result_copy['avg_relevance_score'], np.ndarray):
                    result_copy['avg_relevance_score'] = float(result_copy['avg_relevance_score'])
                results_copy.append(result_copy)

            json.dump(results_copy, f, indent=2, default=str)

        # Create and save summary CSV
        summary_df = self.create_summary_dataframe()
        summary_path = os.path.join(output_dir, "validation_summary.csv")
        summary_df.to_csv(summary_path, index=False)

        print(f"Saved detailed results to {detailed_path}")
        print(f"Saved summary to {summary_path}")

    def analyze_validation_results(self):
        """
        Analyze validation results to identify patterns and issues.

        Returns:
            Dictionary with analysis results
        """
        if not self.validation_results:
            return {"error": "No validation results to analyze"}

        # Collect metrics across all tests
        all_flagged_terms = []
        all_relevance_scores = []
        query_times = []
        coherence_scores = []

        for result in self.validation_results:
            query_times.append(result['query_time'])
            coherence_scores.append(result['coherence']['coherence_score'])

            # Collect all flagged terms
            for validated in result['validated_results']:
                if validated['is_flagged']:
                    all_flagged_terms.extend(validated['matched_terms'])

            # Collect relevance scores
            for validated in result['validated_results']:
                all_relevance_scores.append(validated['relevance']['ingredients_matched_pct'])

        # Analyze flagged content
        flagged_term_counts = Counter(all_flagged_terms)
        most_common_flagged = flagged_term_counts.most_common(5)

        # Calculate average metrics safely
        avg_query_time = 0
        avg_relevance_score = 0
        avg_coherence_score = 0

        if query_times:
            avg_query_time = sum(query_times) / len(query_times)
        if all_relevance_scores:
            avg_relevance_score = sum(all_relevance_scores) / len(all_relevance_scores)
        if coherence_scores:
            avg_coherence_score = sum(coherence_scores) / len(coherence_scores)

        # Overall statistics
        analysis = {
            'test_count': len(self.validation_results),
            'avg_query_time': avg_query_time,
            'avg_relevance_score': avg_relevance_score,
            'avg_coherence_score': avg_coherence_score,
            'flagged_content': {
                'total_flags': sum(flagged_term_counts.values()),
                'unique_terms': len(flagged_term_counts),
                'most_common': most_common_flagged
            }
        }

        # Find best and worst performing tests
        if self.validation_results:
            # Sort by average relevance
            sorted_results = sorted(self.validation_results,
                                    key=lambda x: x['avg_relevance_score'],
                                    reverse=True)

            analysis['best_performing'] = {
                'test_name': sorted_results[0]['test_name'],
                'ingredients': sorted_results[0]['ingredients'],
                'relevance_score': sorted_results[0]['avg_relevance_score']
            }

            analysis['worst_performing'] = {
                'test_name': sorted_results[-1]['test_name'],
                'ingredients': sorted_results[-1]['ingredients'],
                'relevance_score': sorted_results[-1]['avg_relevance_score']
            }

        return analysis


def main():
    # k: Number of recommendations to request
    k = 10

    # test_selection: List of test names to run, or None for all
    test_selection = None

    print("Recipe Recommendation Validation System")
    print("=======================================")

    # Initialize validator
    validator = RecipeValidator(
        ingredients_file="validation/test_ingredients.json",
        inappropriate_terms_file="validation/inappropriate_terms.txt")

    # Load recommender model
    model_loaded = validator.load_recommender("models/recipes_faiss.index", "models/recipes_dataframe.csv")

    if not model_loaded:
        print("Failed to load recommender model.")
        return

    # Run validation
    summary_df = validator.run_validation(test_selection=test_selection, k=k)
    print("\nValidation Summary:")
    print(summary_df)
    validator.save_validation_results()

    # Show analysis
    analysis = validator.analyze_validation_results()
    print("\nValidation Analysis:")
    print(json.dumps(analysis, indent=2))

    return validator

if __name__ == "__main__":
    main()