from visualizer import Visualizer

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class LearningCurveVisualizer(Visualizer):

    @property
    def learning_curves(self):
        if self.num_exps == 0:
            return

        nrows = np.ceil(self.num_exps / 3.).astype(np.int)
        scaling_factor = 1 if self.num_exps > 1 else 2
        fig, axes_matrix = plt.subplots(nrows, ncols=3, figsize=[16, self.num_exps*scaling_factor])
        
        for exp_id, axes in zip(self.exp_ids, axes_matrix.flatten()):
            self._plot_learning_curve(exp_id, axes)
            
        fig.subplots_adjust(wspace=.35, hspace=.65)
            
    def _plot_learning_curve(self, exp_id, axes):
        df = self.folded_dfs[exp_id]

        if self.ensemble:
            cols = ['main_loss', 'ensemble_loss', 'val_main_loss', 'val_ensemble_loss']
        else:
            cols = ['main_loss', 'val_main_loss']
        df[cols].plot(title=str(exp_id), ax=axes, legend=exp_id==self.dummy_id)
        axes.set_ylabel('loss')
        self._mark_best_epoch(exp_id, axes)

    @property
    def validation_curves(self):
        if self.num_exps == 0:
            return

        nrows = np.ceil(self.num_exps / 3.).astype(np.int)
        scaling_factor = 3 if self.num_exps > 1 else 9
        fig, axes_matrix = plt.subplots(3*nrows, ncols=3, figsize=[16, self.num_exps*scaling_factor])
        
        k = 0
        for j in range(nrows):
            for i in range(3):
                if k == self.num_exps:
                    break

                self._plot_validation_curve(self.exp_ids[k], axes_matrix.T[i, j*3:(j+1)*3])

                k += 1

        fig.subplots_adjust(wspace=.35, hspace=.65)
        
    def _plot_validation_curve(self, exp_id, axes_list):
        df = self.folded_dfs[exp_id]
        
        phases = ['{}: train :{}'.format(exp_id, exp_id), '{}: validation :{}'.format(exp_id, exp_id)]

        for i, (phase, prefix, axes) in enumerate(zip(['train', 'validation'], ['main', 'val_main'], axes_list[:2])):
            acc_cols = ['{}_acc_{}'.format(prefix, j) for j in range(7)]
            df[acc_cols].plot(ax=axes, title=phases[i], legend=exp_id==self.dummy_id and i==0)
            if exp_id == self.dummy_id and i == 0:
                lines, labels = axes.get_legend_handles_labels()
                axes.legend(lines, pd.read_csv('classes.csv', index_col=0).label, loc='center')

            axes.set_ylim([0, 1])
            axes.set_ylabel(self.metric)
            self._mark_best_epoch(exp_id, axes)
            
        axes = axes_list[-1]
        if self.ensemble:
            names = ['main', 'ensemble', 'val_main', 'val_ensemble']
        else:
            names = ['main', 'val_main']
        cols = ['{}_{}'.format(s, self.metric) for s in names]
        df[cols].plot(ax=axes, legend=exp_id==self.dummy_id, title='{}: micro :{}'.format(exp_id, exp_id))
        axes.set_ylabel(self.metric)
        # axes.set_ylim([0, 1])
        self._mark_best_epoch(exp_id, axes_list[-1])

    def _mark_best_epoch(self, exp_id, axes):
        df = self.folded_dfs[exp_id]

        best_epoch = self.best_epochs[exp_id]
        axes.axvline(x=best_epoch, c='grey')
