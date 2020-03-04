from sensemakingspace.state import State
import copy
import numpy as np

def random_year(year_range, seed=None):
    min_year, max_year = year_range
    return np.random.randint(min_year, max_year + 1)

def random_countries(all_countries, num_selected=None, seed=None):
    num_all = len(all_countries)
    if seed is not None:
        np.random.seed(seed)
    if num_selected is None:
        num_selected = np.random.randint(num_all + 1)
    return list(np.random.choice(all_countries, num_selected, replace=False))

def random_choice(options, exclude=[], seed=None):
    if seed is not None:
        np.random.seed(seed)
    allowed = list(set(options) - set(exclude))
    choice = np.random.choice(np.array(allowed))
    return choice

def random_axis(exclude=[], seed=None):
    options = [
        'child_mortality',
        'fertility',
        'gdp',
        'life_expect',
        'population'
    ]
    return random_choice(options, exclude=exclude, seed=seed)

def random_color(exclude=[], seed=None):
    options = [
        'continent',
        'main_religion'
    ]
    return random_choice(options, exclude=exclude, seed=seed)

def random_state(year_range, all_countries, num_selected=None, exclude={}, seed={}):
    year = random_year(year_range, seed.get('year', None))
    x = random_axis(exclude.get('x', []), seed.get('x', None))
    y = random_axis(exclude.get('y', []), seed.get('y', None))
    size = random_axis(exclude.get('axis', []), seed.get('axis', None))
    color = random_color(exclude.get('color', []), seed.get('color', None))
    countries = random_countries(all_countries, num_selected, seed.get('countries', None))
    state = State({
        'year': year,
        'x': x,
        'y': y,
        'size': size,
        'color': color,
        'countries': countries
    })    
    return state

"""Functions for creating altered states based on existing ones:"""

def change_year_random(state, year_range, seed=None):
    new_state = copy.deepcopy(state)
    min_year, max_year = year_range
    if seed is not None:
        np.random.seed(seed)
    new_year = np.random.randint(min_year, max_year + 1)
    while(new_year == state.year):
        new_year = np.random.randint(min_year, max_year + 1)
    new_state.year = new_year
    return new_state

def add_to_year(state, add):
    new_state = copy.deepcopy(state)
    new_state.year += add
    return new_state

def change_axis(state, axis, exclude=[], seed=None):
    ex = exclude.copy()
    ex.append(state.__dict__[axis])
    new_value = random_axis(exclude=ex, seed=seed)
    new_state = copy.deepcopy(state)
    new_state.__dict__[axis] = new_value
    return new_state

def change_color(state, exclude=[], seed=None):
    ex = exclude.copy()
    ex.append(state.color)
    new_value = random_color(exclude=ex, seed=seed)
    new_state = copy.deepcopy(state)
    new_state.color = new_value
    return new_state

def add_countries_random(state, add_num, all_countries, seed=None):
    if seed is not None:
        np.random.seed(seed)
    allowed = list(set(all_countries) - set(state.countries))
    new_countries = np.random.choice(allowed, add_num, replace=False)
    new_state = copy.deepcopy(state)
    new_state.countries = list(np.concatenate([state.countries, new_countries]))
    return new_state

def remove_countries_random(state, rmv_num, seed=None):
    if seed is not None:
        np.random.seed(seed)
    rmv_num = min(rmv_num, len(state.countries))
    removed_countries = np.random.choice(state.countries, rmv_num, replace=False)
    new_state = copy.deepcopy(state)
    new_state.countries = list(set(state.countries) - set(removed_countries))
    return new_state