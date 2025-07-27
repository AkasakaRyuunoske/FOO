import os
import pickle
import numpy as np

from django.shortcuts import render, redirect
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from TagManager.models import Tag, RecipeTag
from RecipeManager.models import Recipe

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'TagManager/mvp_tagging/classifiers')

# Carica tokenizer pickle
with open(os.path.join(MODEL_DIR, 'difficulty', 'difficulty_classifier_tokenizer.pkl'), 'rb') as f:
    difficulty_tokenizer = pickle.load(f)

with open(os.path.join(MODEL_DIR, 'prep_time', 'tokenizer.pkl'), 'rb') as f:
    time_tokenizer = pickle.load(f)

# Carica modelli
difficulty_model = load_model(os.path.join(MODEL_DIR, 'difficulty', 'difficulty_classifier.h5'))
time_model = load_model(os.path.join(MODEL_DIR, 'prep_time', 'prep_time_classifier.h5'))

MAX_LEN = 500  # o quello usato nel training

def preprocess_text(text, tokenizer):
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=MAX_LEN)
    return padded

def get_difficulty_tag(predicted_label):
    return Tag.objects.filter(type='Difficulty', name__iexact=predicted_label).first()

def get_time_tag(predicted_time):
    time_tags = Tag.objects.filter(type="tempo")
    for tag in time_tags:
        try:
            range_str = tag.name.replace(" min", "").strip()
            low, high = map(int, range_str.split('-'))
            if low <= predicted_time <= high:
                return tag
        except Exception:
            continue
    return None

def create_recipe(request):
    if request.method == "POST":
        title = request.POST.get('title', '').strip()
        ingredients = request.POST.get('ingredients', '').strip()
        instructions = request.POST.get('instructions', '').strip()

        recipe = Recipe.objects.create(
            title=title,
            ingredients=ingredients,
            instructions=instructions,
        )

        data = ingredients + " " + instructions

        # Preprocess testo per ogni tokenizer separato
        X_time = preprocess_text(data, time_tokenizer)
        X_diff = preprocess_text(data, difficulty_tokenizer)

        # Predizione tempo
        pred_time_val = time_model.predict(X_time)[0][0]
        predicted_time = int(round(pred_time_val))

        # Predizione difficoltà
        diff_pred = difficulty_model.predict(X_diff)[0]

        # Associa tag tempo
        time_tag = get_time_tag(predicted_time)
        if time_tag:
            RecipeTag.objects.create(recipe=recipe, tag=time_tag)

        # Associa tag difficoltà
        diff_tag = get_difficulty_tag(diff_pred)
        if diff_tag:
            RecipeTag.objects.create(recipe=recipe, tag=diff_tag)

        return redirect("recipe_detail", recipe_id=recipe.id)

    return render(request, "new_recipe.html")
