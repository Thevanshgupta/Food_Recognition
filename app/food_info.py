food_database = {
    "pasta": {
        "origin": "Italy",
        "ingredients": ["Pasta", "Tomato Sauce", "Garlic", "Olive Oil"],
        "recipe": "Boil pasta, prepare sauce with garlic and tomato, mix and serve."
    },
    "sushi": {
        "origin": "Japan",
        "ingredients": ["Rice", "Fish", "Seaweed", "Soy Sauce"],
        "recipe": "Roll rice and fish with seaweed, slice, and serve with soy sauce."
    }
}

def get_food_info(food_name):
    return food_database.get(food_name.lower(), {
        "origin": "Unknown",
        "ingredients": [],
        "recipe": "No recipe available."
    })
