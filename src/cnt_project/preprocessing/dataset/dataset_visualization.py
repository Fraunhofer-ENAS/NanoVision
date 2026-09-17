import numpy as np
import os
import matplotlib
import colorsys
import matplotlib.pyplot as plt



def random_label_cmap(n=2**16, h = (0,1), l = (.4,1), s =(.2,.8)):
    
    # cols = np.random.rand(n,3)
    # cols = np.random.uniform(0.1,1.0,(n,3))
    h,l,s = np.random.uniform(*h,n), np.random.uniform(*l,n), np.random.uniform(*s,n)
    cols = np.stack([colorsys.hls_to_rgb(_h,_l,_s) for _h,_l,_s in zip(h,l,s)],axis=0)
    cols[0] = 0
    return matplotlib.colors.ListedColormap(cols)


def plot_img_label(self, lbl_title, save_folder=None):
        lbl_cmap = random_label_cmap()
        
        fig, (ax_img, ax_lbl) = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw={'width_ratios': (1.25, 1)})
        
        img_cmap =  'gray'
        ax_img.imshow(self.X, cmap=img_cmap, clim=(0, 1))
        ax_img.set_title('image')
        ax_img.axis('off')
        ax_img.grid(False)
        #fig.colorbar(im, ax=ax_img)
        
        if lbl_title == "GT_LABEL" and self.Y is not None:
            ax_lbl.imshow(self.Y, cmap=lbl_cmap, interpolation='none')
            ax_lbl.set_title(lbl_title)
            ax_lbl.axis('off')
            ax_lbl.grid(False)

        if lbl_title == "PRED_LABEL" and self.Y_pred is not None:
            ax_lbl.imshow(self.Y_pred, cmap=lbl_cmap, interpolation='none')
            ax_lbl.set_title(lbl_title)
            ax_lbl.axis('off')
            ax_lbl.grid(False)
            
        
        plt.tight_layout()
        
        if save_folder is not None:
            os.makedirs(save_folder, exist_ok=True)
            new_save_path = os.path.join(save_folder, f"{self.X_fn}_{lbl_title}.png")
            plt.savefig(new_save_path, dpi=300)
            plt.close(fig)
        else:
            plt.show()