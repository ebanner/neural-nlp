import os

import numpy as np
import pandas as pd


class Visualizer:
    
    def __init__(self, exp_group, da_folds=None, metric='acc'):
        """Collect experiment ids and process dataframes

        Also add a dummy dataframe so we can view the legend.

        """
        # collect experiment ids
        self.exp_group = exp_group
        self.exp_ids = get_ipython().getoutput(u'ls -v store/train/$exp_group')
        self.exp_ids = [exp_id for exp_id in self.exp_ids if os.listdir('store/train/{}/{}'.format(exp_group, exp_id))]
        self.exp_ids = [int(exp_id) for exp_id in self.exp_ids]
        self.num_exps = len(self.exp_ids)
        self.metric = metric

        self.dfs = {}
        for exp_id in self.exp_ids:
            # concatenate data from all folds into one dataframe
            folds = get_ipython().getoutput(u'ls -v store/train/$exp_group/$exp_id') if not da_folds else da_folds
            folds = [int(fold.split('.')[0]) for fold in folds if type(fold) == str] if not da_folds else da_folds

            df = pd.DataFrame()
            for fold in folds:
                dff = pd.read_csv('store/train/{}/{}/{}.csv'.format(self.exp_group, exp_id, fold)).fillna(0)
                dff['fold'] = fold # attach fold so we can identify it
                df = pd.concat([df, dff])

            df.name = exp_id
            self.dfs[exp_id] = df

        # check for ensembling
        df = pd.read_csv('store/hyperparams/{}/{}.csv'.format(self.exp_group, self.exp_ids[0]), index_col=0)
        self.ensemble = 'ensemble-ids' in df.index

        self.folded_dfs = {}
        for exp_id, df in self.dfs.items():
            # average across folds
            self.folded_dfs[exp_id] = self._process_cv(df)

        # add a dummy experiment group for visualization
        tmp_id = self.exp_ids[-1]
        self.dummy_id = tmp_id + 1
        self.exp_ids += [self.dummy_id]
        df = self.dfs[0].copy()
        dff = pd.DataFrame(np.zeros_like(df), columns=df.columns)
        dff.name = self.dummy_id
        dff['exp-id'] = self.dummy_id # get the all-important fold information
        self.dfs[self.dummy_id] = dff
        self.folded_dfs[self.dummy_id] = self._process_cv(dff)
        self.num_exps += 1
        df = pd.read_csv('store/hyperparams/{}/{}.csv'.format(self.exp_group, tmp_id), index_col=0)
        df.ix['exp-id'] = self.dummy_id
        df.to_csv('store/hyperparams/{}/{}.csv'.format(self.exp_group, self.dummy_id))
      
    def _process_cv(self, df):
        """Average performance across folds

        Due to early stopping, different folds may have a different number of
        epochs. One step to be able to consistently average folds is to force
        all runs to have the same number of epochs. In order to do this, we take
        folds which have a smaller number of epochs and repeat the last row
        until it has the same length as the run with the highest number of
        epochs.

        """
        # average across folds
        exp_id = df.name
        groups = [group for i, group in df.groupby('fold')]
        max_nb_epoch = max(len(group) for group in groups)

        extended_dfs = [0]*len(groups) # force each fold to be the same number of epochs
        for i, group in enumerate(groups):
            fixup_len = max_nb_epoch-len(group)
            if fixup_len > 0:
                extended_dfs[i] = pd.concat([group, pd.concat([group.iloc[-1:]]*fixup_len)])
                extended_dfs[i].index = range(max_nb_epoch) # fixup index
            else:
                extended_dfs[i] = group

        averaged = sum(extended_dfs) / len(groups) # average across folds
        columns = df.columns.tolist()
        columns.remove('fold')
        df = pd.DataFrame(averaged, columns=columns) # don't include the `fold` column
        df.name = exp_id

        # get best accs from each fold
        output_types = ['main']
        output_types += ['ensemble'] if self.ensemble else []
        for output_type in output_types:
            score_col = 'val_{}_{}'.format(output_type, self.metric)
            best_accs = [group[score_col].max() for group in groups]

            # record stats about these best scores
            df['best_{}_mu'.format(output_type)] = np.mean(best_accs)
            df['best_{}_range'.format(output_type)] = np.ptp(best_accs)
            df['best_{}'.format(output_type)] = np.max(best_accs) 
            df['worst_{}'.format(output_type)] = np.min(best_accs)

        return df

    def best_runs(self, ascending=False):
        if self.num_exps == 0:
            return

        df = pd.DataFrame()
        self.best_epochs, score_col = {}, 'val_{}_{}'.format('ensemble' if self.ensemble else 'main', self.metric)

        # filter down columns to display
        disp_cols = self.folded_dfs.values()[0]
        disp_cols = [col for col in disp_cols if 'best' in col or 'worst' in col or col == score_col]

        for exp_id, cv_df in self.folded_dfs.items():
            max_idx = self.folded_dfs[exp_id][score_col].argmax()
            dff = cv_df.ix[[max_idx]][disp_cols]

            # tack on additional visual info
            dff['best_epoch'] = self.best_epochs[exp_id] = max_idx
            hp_df = pd.read_csv('store/hyperparams/{}/{}.csv'.format(self.exp_group, exp_id), index_col=0).T
            hp_df.index = dff.index
            dff = pd.concat([dff, hp_df], axis=1)

            df = pd.concat([df, dff])

        os.remove('store/hyperparams/{}/{}.csv'.format(self.exp_group, self.dummy_id)) # get rid of dummy hyperparams
            
        sort_col = 'best_{}_mu'.format('ensemble' if self.ensemble else 'main')
        sorted_df = df.sort_values(by=sort_col, ascending=ascending).set_index(keys='exp-id')
        self.exp_ids = [int(exp_id) for exp_id in sorted_df.index]
        self.sorted_df = sorted_df[sorted_df.index != str(self.dummy_id)] # don't show dummy id

        return self.sorted_df
