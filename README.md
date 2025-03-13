	FOO => Food Oracle and Overseer 

**FOO (Food Oracle and Overseer)** is a web-based, ML-powered recipe recommendation platform designed to simplify and personalize meal selection. Users effortlessly discover recipes tailored to their preferences, ingredients availability, and cooking style through intelligent, machine-learning-driven recommendations.

**FOO** features two main intelligent modules:

- **Automatic Multi-tag classification** (Difficulty, Cooking Time, Cost, Method) powered by **XGBoost**, ensuring accurate and consistent labeling of recipes.
- **Intelligent Ingredient-based Recommendations** using advanced word embeddings (Word2Vec), enabling users to creatively use available ingredients.

Other core functionalities include user authentication, favorite recipes management, automatic unit conversion to metric, and intuitive filtering/search capabilities. Built on a modern stack (Django, PostgreSQL, HTMX, Docker), FOO emphasizes portability, maintainability, documentation, and user-centric design.

Tech stack
=============================
	
FE $->$ HTMX, CSS, JSON, Chart library(any) <br>
BE $->$ Python, Django, Pandas, Swagger, nltk  <br>
DB $->$ PostgreSQL  <br>
Tools $->$ Docker  <br>
Tagging $->$ Random Forest / XGBoost  <br>
Embedding-based Recipe Prediction $->$ Word Embedding: Word2Vec / Transformers  <br>
Security $->$ JWT?  <br>
Arch $->$ API First Approach, REST(json)  <br>

Goals
==============================
1. Portability. Application must be very easy to use and deploy.
2. Documentation. Application must be documented, tested and designed.
3. 2 Main ml modules: Classification for tags and Embedding for recipe prediction.

Features
==============================
1. User management. Manage Authentication allowing each user to have a personal list of favorite dishes.
2. CRUD operations for user-recipe. Allow user's to create a personal recipe inputting: title, ingredients and steps. Update existing recipes or delete those unwanted.
3. Automatic tagging when adding/updating a recipe based on ingredients, title and steps.
4. Offer recipes based on:
	1. Random selection (recipe of the day)
	2. User inputs tags like: quick to cook, cheap, healthy, sweet
	3. User inputs ingredients like: eggs, chicken, garlic, bread
5. Offer a hide/show feature: user can hide for X period of time a dish, or hide it completely, so it won't be offered again.
