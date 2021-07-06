
from .story import Story
from .utils import graphplot
import re
import json
import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate
from scipy.spatial.distance import pdist, squareform
import pandas as pd
import copy
from numba import njit, float64
from numba import jit
from openTSNE import TSNE as openTSNE
from openTSNE.callbacks import ErrorLogger
from sklearn.manifold import TSNE as sklearnTSNE
from sklearn.manifold import MDS
from umap import UMAP
from csv import QUOTE_NONNUMERIC
import networkx as nx
import pandas as pd
from datashader.bundling import connect_edges
from datashader.layout import forceatlas2_layout
from datashader.bundling import connect_edges, hammer_bundle

class Stories:
    
    def __init__(self, source, drive_mounted=True, uploaded=None, names=None, set_initial=True):
        self.projected = False
        self.stories = []
        self.counts = None

        if names is not None:
            assert len(names) == len(source), 'Length of "names" must match number of stories!'
        for idx, s in enumerate(source):
            if isinstance(s, Story):
                if names is not None:
                    s.set_name(names[idx])
                self.stories.append(s)
            elif isinstance(s, str):
                if names is not None:
                    name = names[idx]
                else:
                    name = None
                story = Story(s, drive_mounted=True, uploaded=None, name=name, set_initial=True)
                self.stories.append(story)
                
        if names is None:
            self.names = [s.name for s in self.stories]
        else:
            self.names = names
            
    def __len__(self):
        return len(self.stories)

    def __str__(self):
        pstr = 'Stories(\n'
        for st in self.stories:
            pstr+= '\t{},\n'.format(st.__str__())
        return pstr[:-2] + '\n)'

    def __repr__(self):
        if len(self) == 1:
            return 'Stories(<1 Story>)'
        else:
            return 'Story(<{} Stories>)'.format(len(self))

    def __getitem__(self, key):
        return self.stories[key]
            
    def lengths(self):
        return [len(s) for s in self.stories]
    
    def min_year(self):
        return np.array([s.min_year() for s in self.stories]).min()
    
    def max_year(self):
        return np.array([s.max_year() for s in self.stories]).max()
    
    def year_range(self):
        return (self.min_year(), self.max_year())

    def year_span(self):
        return np.abs(self.max_year() - self.min_year())
    
    def countries(self):
        return np.unique(np.concatenate([s.countries() for s in self.stories]))
    
    def encode(self, flat=True, countries=None, condense_countries=False):
        if countries is None:
            countries = self.countries()
        
        if flat:
            return np.concatenate([s.encode(
                countries,
                condense_countries=condense_countries) for s in self.stories])
        else:
            return np.stack([s.encode(
                countries,
                condense_countries=condense_countries) for s in self.stories])
    
    def project(self, weights={}, alpha=.5, verbose=False, delete_duplicates=False, method='tsne', implementation='openTSNE', condense_countries=False, **kwargs):        
        w_dict = {
            'year': 1.,
            'x': 1.,
            'y': 1.,
            'size': 1.,
            'color': 1.,
            'countries': 1.
        }        
        for w in weights:
            w_dict[w] = weights[w]            
        weights = np.array(list(w_dict.values()), dtype=np.float32)
        year_span = self.year_span()
        num_countries = len(self.countries())

        if not condense_countries:
            @jit(nopython=True)
            def state_distance(a, b):
                a = a.astype(np.float32)
                b = b.astype(np.float32)
                if year_span == 0:
                    year_dist = 0
                else:
                    year_dist = np.abs(a[0] - b[0]) / year_span
                x_dist = 1 - np.dot(a[1:6], b[1:6])
                y_dist = 1 - np.dot(a[6:11], b[6:11])
                size_dist = 1 - np.dot(a[11:16], b[11:16])
                color_dist = 1 - np.dot(a[16:18], b[16:18])
                if num_countries == 0:
                    country_dist = 0
                else:
                    country_dist = np.linalg.norm(a[18:] - b[18:]) / np.sqrt(num_countries)
            
                dists = np.array([year_dist, x_dist, y_dist, size_dist, color_dist, country_dist], dtype=np.float32)
            
                return (dists*weights).sum()
        else:
            @jit(nopython=True)
            def state_distance(a, b):
                a = a.astype(np.float32)
                b = b.astype(np.float32)
                if year_span == 0:
                    year_dist = 0
                else:
                    year_dist = np.abs(a[0] - b[0]) / year_span
                x_dist = 1 - np.dot(a[1:6], b[1:6])
                y_dist = 1 - np.dot(a[6:11], b[6:11])
                size_dist = 1 - np.dot(a[11:16], b[11:16])
                color_dist = 1 - np.dot(a[16:18], b[16:18])
                if num_countries == 0:
                    country_dist = 0
                else:
                    lat1, long1 = a[18:20]
                    lat2, long2 = b[18:20]
                    lat1 = lat1 / 180. * 2 * np.pi
                    lat2 = lat2 / 180. * 2 * np.pi
                    delta_long = long1 - long2
                    gcdist = np.sin(lat1) * np.sin(lat2)
                    gcdist += np.cos(lat1) * np.cos(lat2) * np.cos(delta_long)
                    gcdist = np.arccos(gcdist) / np.pi
                    
                    spread_dist = np.abs(a[20] - b[20])
                    
                    sel_dist = np.abs(a[21] - b[21]) / num_countries

                    country_dist = (gcdist + spread_dist + sel_dist) 
            
                dists = np.array([year_dist, x_dist, y_dist, size_dist, color_dist, country_dist], dtype=np.float32)
            
                return (dists*weights).sum()
        
        if delete_duplicates:
            # check if any weight is 0.
            encoded = self.encode()
            if w_dict['year'] == 0.:
                encoded[:,0] = 0.
            if w_dict['x'] == 0.:
                encoded[:,1:6] = 0.
            if w_dict['y'] == 0.:
                encoded[:,6:11] = 0.
            if w_dict['size'] == 0.:
                encoded[:,11:16] = 0.
            if w_dict['color'] == 0.:
                encoded[:,16:18] = 0.
            if w_dict['countries'] == 0.:
                encoded[:,18:] = 0.
            encoded, indices, counts = np.unique(encoded, axis=0, return_inverse=True, return_counts=True)
            self.counts = counts[indices]
        else:
            encoded = self.encode()

        if method == 'tsne':
            if implementation == 'openTSNE':
                if verbose:
                    tsne = openTSNE(
                        metric=state_distance,
                        callbacks=ErrorLogger(),
                        n_jobs=-1,
                        **kwargs
                    )
                else:
                    tsne = openTSNE(
                        metric=state_distance,
                        n_jobs=-1,
                        **kwargs
                    )
                embedding = np.array(tsne.fit(encoded))
            elif implementation == 'sklearn':
                tsne = sklearnTSNE(
                    metric=state_distance,
                    verbose=3 if verbose else 0)
                embedding = np.array(tsne.fit_transform(encoded))
        elif method == 'mds':
            mds = MDS(
                n_components=2,
                metric=True,
                dissimilarity='precomputed'
            )
            distmat = squareform(pdist(encoded, state_distance))
            embedding = mds.fit_transform(distmat)
        elif method == 'umap':
            umap = UMAP(
                metric=state_distance,
                verbose=verbose,
                **kwargs)
            embedding = np.array(umap.fit_transform(encoded))
        elif method == 'hybrid':
            if not delete_duplicates:
                raise Warning('Hybrid layout always deletes duplicates!')

            adj = nx.adj_matrix(self.make_graph()).toarray()

            # adjacency matrix of undirected multigraph
            intermediate = np.zeros_like(adj, dtype=np.int)
            for (i, j), item in np.ndenumerate(adj):
                intermediate[i,j] += item
                intermediate[j,i] += item

            # use inverse number of connections as weights
            edges = []
            weights = []
            for (i,j), item in np.ndenumerate(intermediate):
                if item != 0 and i <= j:
                    edges.append((i,j))
                    weights.append(item)

            # construct weighted graph and calculate path lengths
            g = nx.Graph()
            for e, w in zip(edges, weights):
                g.add_edge(*e, weight=1/w)
            path_lengths = dict(nx.all_pairs_dijkstra_path_length(g))

            # construct distmat from path_lengths
            graph_distmat = np.zeros_like(adj, dtype=np.float)
            graph_distmat -= np.inf
            for i in path_lengths:
                dists = path_lengths[i]
                for j in dists:
                    graph_distmat[i,j] = dists[j]

            graph_distmat = graph_distmat / graph_distmat.max()

            attr_distmat = squareform(pdist(
                np.unique(self.encode().reshape(-1,100), axis=0),
                metric=state_distance
                )
            )

            init = 'random'
            if hasattr(alpha, '__iter__'):
                self.hybrid_alphas = alpha
                embedding = []
                for a in alpha:
                    distmat = (1 - a) * graph_distmat + a * attr_distmat
                    tsne = openTSNE(metric='precomputed', initialization=init)
                    embedding.append(tsne.fit(distmat))
            else:
                distmat = (1 - alpha) * graph_distmat + alpha * attr_distmat
                tsne = openTSNE(metric='precomputed', initialization=init)
                embedding = tsne.fit(distmat)
                init = embedding[-1]

        if delete_duplicates:
            if method == 'hybrid' and hasattr(alpha, '__iter__'):
                embedding = np.stack([ e[indices] for e in embedding])
            else:
                embedding = embedding[indices]
        
        indices = np.add.accumulate(self.lengths())
        self.embedding = np.array_split(embedding, indices)[:-1]
        
        self.projected = True
    
    def plot(self):
        if not self.projected:
            raise Warning('Run projection first!')
        fig = plt.figure(figsize=(15,15))
        ax = fig.add_subplot(111)
        for idx, story in enumerate(self.embedding):
            tck, u = interpolate.splprep((story + (1e-6*np.random.rand(*(story.shape)))).transpose(), s=0)
            unew = np.arange(0, 1.01, 0.001)
            out = interpolate.splev(unew, tck)
            ax.plot(out[0], out[1], alpha=0.7, label=self.names[idx])
            ax.scatter(story[:,0], story[:,1], s=20)
            ax.scatter(story[:1,0], story[:1,1], s=50, color='black')
        plt.legend()

    def make_graph(self):
        encoded, indices, counts = np.unique(self.encode(), axis=0, return_inverse=True, return_counts=True)
        leni = np.add.accumulate(self.lengths())
        edges = np.stack([
                np.delete(indices, leni-1),
                np.delete(indices, leni)[1:]
            ]).transpose()
        digr = nx.MultiDiGraph()
        digr.add_edges_from(edges);
        return digr

    def plot_graph(self, method='networkx', options={}):
        digr = self.make_graph()
        if method == 'networkx':
            dflt_options = {
                'node_color': 'black',
                'node_size': 20,
                'width': 1,
            }
            opts = {**dflt_options, **options}
            nx.draw_kamada_kawai(digr, **opts)
        elif method == 'datashader':
            digr = digr.to_undirected()
            nodes = pd.DataFrame(digr.nodes(), columns=['name'])
            edges = pd.DataFrame(digr.edges(), columns=['source', 'target'])
            iterations = {'iterations': int(np.ceil(np.sqrt(len(nodes)))), **options}['iterations']
            fd = forceatlas2_layout(nodes, edges, iterations=iterations)
            bundle = {'bundle': False, **options}['bundle']
            if bundle:
                return graphplot(fd, hammer_bundle(fd,edges))
            else:
                return graphplot(fd, connect_edges(fd,edges))

    
    def export_csv(self, filename, ids=None, list_changes=False):
        dframe = pd.DataFrame()
        if not self.projected:
            print('Saving CSV without embedding!')
        elif type(self.embedding[0]) is list:
            multiple_embeddings = True
        else:
            multiple_embeddings = False

        names = np.repeat(self.names, self.lengths())
        if ids is None:
            ids = np.arange(len(self))
        path_idx = np.repeat(ids, self.lengths())
        data = np.concatenate([[[s.x, s.y, s.size, s.color, s.year] for s in st.states] for st in self])
        this_x, this_y, this_size, this_color, this_year, = data.transpose()
        this_countries = np.concatenate([[s.country_string() for s in st.states] for st in self])

        dframe['line'] = path_idx
        dframe['algo'] = names
        dframe['new_x'] = this_x
        dframe['new_y'] = this_y
        dframe['new_size'] = this_size
        dframe['new_color'] = this_color
        dframe['new_year'] = this_year
        dframe['new_country'] = this_countries
        dframe['old_x'] = np.insert(this_x[:-1], 0, this_x[0])
        dframe['old_y'] = np.insert(this_y[:-1], 0, this_y[0])
        dframe['old_size'] = np.insert(this_size[:-1], 0, this_size[0])
        dframe['old_color'] = np.insert(this_color[:-1], 0, this_color[0])
        dframe['old_year'] = np.insert(this_year[:-1], 0, this_year[0])
        dframe['old_country'] = np.insert(this_countries[:-1], 0, this_countries[0])

        if list_changes:
            changes = np.concatenate([st.change_string_list() for st in self])
            dframe['changes'] = changes

        if self.counts is not None:
            mult_label = 'multiplicity[{m_min};{m_max}]'.format(
                m_min=self.counts.min(),
                m_max=self.counts.max()
            ) 
            dframe[mult_label] = self.counts

        if multiple_embeddings:
            for (emb, a) in zip(self.embedding, self.hybrid_alphas):
                x,y = np.concatenate(emb).transpose()
                dframe['x'] = x
                dframe['y'] = y
                basename, extension = filename.split('.')
                dframe.to_csv('{basename}_alpha{alpha:03}.{extension}'.format(
                    basename=basename,
                    alpha=(int(100*a)),
                    extension=extension
                ),
                    index=False,
                    quoting=QUOTE_NONNUMERIC
                )
        else:
            x,y = np.concatenate(self.embedding).transpose()
            dframe['x'] = x
            dframe['y'] = y
            dframe.to_csv(filename,
                    index=False,
                    quoting=QUOTE_NONNUMERIC
                )

    def download_csv(self, filename, ids=None, list_changes=False):
        from google.colab import files
        self.export_csv(filename, ids=ids, list_changes=list_changes)
        files.download(filename)