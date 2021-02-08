
from .state import State
import json
import numpy as np
from openTSNE import TSNE
from openTSNE.callbacks import ErrorLogger
from umap import UMAP
from numba import njit, float64
from numba import jit

initial_state = State({
        'year': 1800,
        'x': 'gdp',
        'y': 'life_expect',
        'size': 'population',
        'color': 'continent',
        'countries': [],
        'timestamp': t
    })

class Story:

    def __init__(self, source=None, drive_mounted=True, uploaded=None, name=None, set_initial=True):

        self.name = name

        # Story can be created from a list of State objects
        #  (possibly handy for artificial stories)
        # or from a JSON file containing a provenance graph

        # if not set_initial:
        #     assert len(source) != 0, 'A story must have at least one state!'

        if type(source) is list and len(source) > 0 and all([isinstance(s, State) for s in source]):
            self.states = source
        elif isinstance(source, str):
            self.source = source
            self.states = self.from_json(
                source, drive_mounted=drive_mounted, uploaded=uploaded)
        elif set_initial:
            self.states = []
        else:
            raise Exception('Story must have at elast one state!')

        # check if first state is initial Gapminder state
        # if not, insert initial Gapminder state

        if set_initial and self.states is not []:
          self.states.insert(0, initial_state(self.states[0].timestamp))

    def from_json(self, filename, drive_mounted=True, uploaded=None):

        # check if Drive Mounted or files were uploaded
        # then parse files accordinlgy

        if not drive_mounted and uploaded is None:
            raise Exception(
                'If Google Drive is not mounted, please upload the input file and pass the resulting dictionary "dict" via "uploaded=dict".')
        if drive_mounted:
            with open(filename) as f:
                json_story = json.load(f)
        else:
            json_story = json.loads(uploaded[filename])

        # extract nodes from provencance graphs
        # then extract visStates from nodes

        states = []
        for node in json_story['nodes']:
            if node['type'] == 'state':
                states.append(node)
        states = [json.loads(s['attrs']['visState']) for s in states[1:]]

        # remove Scagnostics dummy values
        #     (might be interesting to include
        #      if real values are available)

        timestamps = []
        for node in json_story['nodes']:
          if node['type'] == 'action':
            # timestamps.append(node)
            timestamps.append(node['attrs']['meta']['timestamp'])
        
        clean_states = []
        for state in states:
            cleaned = []
            for obj in state:
                if obj['group'] != 'Scatterplot Statistics':
                    cleaned.append(obj)
            clean_states.append(cleaned)
        states = clean_states

        # transform to State objects

        readable_states = []
        for state, t in zip(states, timestamps):
            year = state[0]['payload']['numVal']
            x = state[1]['id']
            y = state[2]['id']
            size = state[3]['id']
            color = state[4]['id']
            # projection = state[5]['id']
            countries = []
            if state[6:] != []:
                for c in state[6:]:
                    countries.append(c['text'])
            readable_states.append(State({
                'year': year,
                'x': x,
                'y': y,
                'size': size,
                'color': color,
                # 'projection': projection,
                'countries': countries,
                'timestamp': t
            }))
        states = readable_states

        return states

    def __len__(self):
        return len(self.states)

    def __str__(self):
        if len(self) == 1:
            return 'Story(<1 State>, name="{name}")'.format(
                name=self.name
            )
        else:
            return 'Story(<{length} States>, name="{name}")'.format(
                length=len(self),
                name=self.name
            )

    def __repr__(self):
        return self.__str__()

    def __getitem__(self, key):
        return self.states[key]

    def append(self, state):
        self.states.append(state)

    def set_name(self, name):
        self.name = name

    def min_year(self):
        return np.array([s.year for s in self.states]).min()

    def max_year(self):
        return np.array([s.year for s in self.states]).max()

    def year_range(self):
        return (self.min_year(), self.max_year())

    def countries(self):
        return np.unique(np.concatenate([s.countries for s in self.states]))

    def encode(self, countries=None, condense_countries=False):
        if countries is None:
            countries = self.countries()

        return np.stack([s.encode(
            countries,
            condense_countries=condense_countries) for s in self.states])

    def change_string_list(self):
        assert len(self) > 2, 'Story must have at least two states!'
        strings = ['None']
        for idx, state in enumerate(self.states[:-1]):
            strings.append(state.change_string(self.states[idx+1]))
        return strings

    def print_changes(self):
        for s in self.change_string_list()[1:]:
            print(s)
