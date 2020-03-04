
import re
import numpy as np
from sensemakingspace.utils import country_selection_to_vector

class State:
    
    def __init__(self, state_dict):
        self.year = state_dict['year']
        self.x = state_dict['x']
        self.y = state_dict['y']
        self.size = state_dict['size']
        self.color = state_dict['color']
        self.countries = state_dict['countries']

    def __str__(self):
        if len(self.countries) == 1:
            cstr = 'country'
        else:
            cstr = 'countries'
        pstr = 'State(\n'
        pstr += '\t"year": {},\n'.format(self.year)
        pstr += '\t"x": {},\n'.format(self.x)
        pstr += '\t"y": {},\n'.format(self.y)
        pstr += '\t"size": {},\n'.format(self.size)
        pstr += '\t"color": {},\n'.format(self.color)
        pstr += '\t"countries": [<{length} {cstr}>]\n)'.format(
            length=len(self.countries), cstr=cstr)
        return pstr

    def __repr__(self):
        return 'State(...)'
        
    def encode_axis(self, attr):
        options = {
            'child_mortality': 0,
            'fertility': 0,
            'gdp': 0,
            'life_expect': 0,
            'population': 0
        }
        options[getattr(self, attr)] = 1
        one_hot = list(options.values())
        return np.array(one_hot)
    
    def encode_color(self):
        options = {
            'continent': 0,
            'main_religion': 0
        }
        options[self.color] = 1
        return np.array(list(options.values()))
        
    def encode_countries(self, countries, condense=False):
        if condense:
            return country_selection_to_vector(countries)
        else:
            cdict = dict([[c,0] for c in countries])
            for c in self.countries:
                cdict[c] = 1
            return np.array(list(cdict.values()))
    
    def encode(self, countries, condense_countries=False):
        nested_vector = [
            np.array([self.year]),
            self.encode_axis('x'),
            self.encode_axis('y'),
            self.encode_axis('size'),
            self.encode_color(),
            self.encode_countries(countries, condense=condense_countries)
        ]
        return np.concatenate(nested_vector)
    
    def country_string(self):
        ctr_list = list(set(self.countries))
        if ctr_list == []:
            return '[]'
        else:
            s = '['
            for ctr in ctr_list[:-1]:
                s += '"{c}", '.format(c=ctr)
            s += '"{c}"]'.format(c=ctr_list[-1])
            return s

    def change_string(self, other, verbose=False):
        for attr in ['year', 'x', 'y', 'size', 'color']:
            if getattr(self, attr) != getattr(other, attr):
                desc = ''
                if attr == 'x' or attr == 'y':
                    desc = '-axis attribute'
                elif attr != 'year':
                    desc = ' attribute'
                change = 'Changed {attr}{desc} from {old} to {new}'.format(
                    attr=attr,
                    desc=desc,
                    old=getattr(self, attr),
                    new=getattr(other, attr)
                )
                return change
        if self.countries != other.countries:
            removed = set(self.countries) - set(other.countries)
            added = set(other.countries) - set(self.countries)
            if len(removed) != 0 and len(added) == 0:
                change = 'Removed {removed} from country selection.'.format(
                    removed=removed
                )
            elif len(removed) == 0 and len(added) != 0:
                change = 'Added {added} to country selection.'.format(
                    added=added
                )
            elif len(removed) != 0 and len(added) != 0:
                change = 'Added {added} to country selection and removed {removed} from country selection'.format(
                    added=added,
                    removed=removed
                )
            change = re.sub(",", ";", change) # replace commas caused by set formatting
            #change = re.sub(',', '', change)    # delete commas remaining in country names
            return change