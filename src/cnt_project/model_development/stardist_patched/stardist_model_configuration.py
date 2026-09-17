from __future__ import annotations

from csbdeep.utils.tf import keras_import, IS_TF_1, CARETensorBoard, CARETensorBoardImage, IS_KERAS_3_PLUS, BACKEND as K
from csbdeep.utils import _raise,  axes_check_and_normalize, axes_dict
from csbdeep.internals.blocks import unet_block
from stardist.models.model2d import Config2D
from cnt_project.model_development.stardist_patched.stardist_preprocessing import MyStarDistData2D
from cnt_project.model_development.stardist_patched.stardist_inference import MyStarDistBase
from cnt_project.model_development.postprocessing.adapters.stardist_postprocess_adapter import postprocess_instances_merge_polygons
from stardist.models.base import  _tf_version_at_least # StarDistBase
from stardist.nms import non_maximum_suppression
import numpy as np
import warnings


# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
# if project_root not in sys.path:
#     sys.path.append(project_root)

Sequence = keras_import('utils', 'Sequence')
Adam = keras_import('optimizers', 'Adam')
ReduceLROnPlateau, TensorBoard = keras_import('callbacks', 'ReduceLROnPlateau', 'TensorBoard')
keras = keras_import()
Input, Conv2D, MaxPooling2D = keras_import('layers', 'Input', 'Conv2D', 'MaxPooling2D')
Model = keras_import('models', 'Model')


# Custom StarDist model (model configuration and training)
class MyStarDist2D(MyStarDistBase):
    """StarDist2D model.

    Parameters
    ----------
    config : :class:`Config` or None
        Will be saved to disk as JSON (``config.json``).
        If set to ``None``, will be loaded from disk (must exist).
    name : str or None
        Model name. Uses a timestamp if set to ``None`` (default).
    basedir : str
        Directory that contains (or will contain) a folder with the given model name.

    Raises
    ------
    FileNotFoundError
        If ``config=None`` and config cannot be loaded from disk.
    ValueError
        Illegal arguments, including invalid configuration.

    Attributes
    ----------
    config : :class:`Config`
        Configuration, as provided during instantiation.
    keras_model : `Keras model <https://keras.io/getting-started/functional-api-guide/>`_
        Keras neural network model.
    name : str
        Model name.
    logdir : :class:`pathlib.Path`
        Path to model folder (which stores configuration, weights, etc.)
    """

    def __init__(self, config=Config2D(), name=None, basedir='.'):
        """See class docstring."""
        super().__init__(config, name=name, basedir=basedir)


    def _build(self):
        self.config.backbone == 'unet' or _raise(NotImplementedError())
        unet_kwargs = {k[len('unet_'):]:v for (k,v) in vars(self.config).items() if k.startswith('unet_')}

        input_img = Input(self.config.net_input_shape, name='input')

        # maxpool input image to grid size
        pooled = np.array([1,1])
        pooled_img = input_img
        while tuple(pooled) != tuple(self.config.grid):
            pool = 1 + (np.asarray(self.config.grid) > pooled)
            pooled *= pool
            for _ in range(self.config.unet_n_conv_per_depth):
                pooled_img = Conv2D(self.config.unet_n_filter_base, self.config.unet_kernel_size,
                                    padding='same', activation=self.config.unet_activation)(pooled_img)
            pooled_img = MaxPooling2D(pool)(pooled_img)

        unet_base = unet_block(**unet_kwargs)(pooled_img)

        if self.config.net_conv_after_unet > 0:
            unet = Conv2D(self.config.net_conv_after_unet, self.config.unet_kernel_size,
                          name='features', padding='same', activation=self.config.unet_activation)(unet_base)
        else:
            unet = unet_base

        output_prob = Conv2D(                 1, (1,1), name='prob', padding='same', activation='sigmoid')(unet)
        output_dist = Conv2D(self.config.n_rays, (1,1), name='dist', padding='same', activation='linear')(unet)

        # attach extra classification head when self.n_classes is given
        if self._is_multiclass():
            if self.config.net_conv_after_unet > 0:
                unet_class  = Conv2D(self.config.net_conv_after_unet, self.config.unet_kernel_size,
                                     name='features_class', padding='same', activation=self.config.unet_activation)(unet_base)
            else:
                unet_class  = unet_base

            output_prob_class  = Conv2D(self.config.n_classes+1, (1,1), name='prob_class', padding='same', activation='softmax')(unet_class)
            return Model([input_img], [output_prob,output_dist,output_prob_class])
        else:
            return Model([input_img], [output_prob,output_dist])


    def train(
        self,
        X,
        Y,
        validation_data,
        classes='auto',
        augmenter=None,
        seed=None,
        epochs=None,
        steps_per_epoch=None,
        workers=1,
        mode="standard",
        debug_plots: bool = False,
        debug_plot_dir: str | None = None,
        debug_run_name: str | None = None,
    ):
        """Train the neural network with the given data.

        Parameters
        ----------
        X : tuple, list, `numpy.ndarray`, `keras.utils.Sequence`
            Input images
        Y : tuple, list, `numpy.ndarray`, `keras.utils.Sequence`
            Label masks
            Positive pixel values denote object instance ids (0 for background).
            Negative values can be used to turn off all losses for the corresponding pixels (e.g. for regions that haven't been labeled).
        classes (optional): 'auto' or iterable of same length as X
             label id -> class id mapping for each label mask of Y if multiclass prediction is activated (n_classes > 0)
             list of dicts with label id -> class id (1,...,n_classes)
             'auto' -> all objects will be assigned to the first non-background class,
                       or will be ignored if config.n_classes is None
        validation_data : tuple(:class:`numpy.ndarray`, :class:`numpy.ndarray`) or triple (if multiclass)
            Tuple (triple if multiclass) of X,Y,[classes] validation data.
        augmenter : None or callable
            Function with expected signature ``xt, yt = augmenter(x, y)``
            that takes in a single pair of input/label image (x,y) and returns
            the transformed images (xt, yt) for the purpose of data augmentation
            during training. Not applied to validation images.
            Example:
            def simple_augmenter(x,y):
                x = x + 0.05*np.random.normal(0,1,x.shape)
                return x,y
        seed : int
            Convenience to set ``np.random.seed(seed)``. (To obtain reproducible validation patches, etc.)
        epochs : int
            Optional argument to use instead of the value from ``config``.
        steps_per_epoch : int
            Optional argument to use instead of the value from ``config``.

        Returns
        -------
        ``History`` object
            See `Keras training history <https://keras.io/models/model/#fit>`_.

        """
        if seed is not None:
            # https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
            np.random.seed(seed)
        if epochs is None:
            epochs = self.config.train_epochs
        if steps_per_epoch is None:
            steps_per_epoch = self.config.train_steps_per_epoch

        classes = self._parse_classes_arg(classes, len(X))

        if not self._is_multiclass() and classes is not None:
            warnings.warn("Ignoring given classes as n_classes is set to None")

        isinstance(validation_data,(list,tuple)) or _raise(ValueError())
        if self._is_multiclass() and len(validation_data) == 2:
            validation_data = tuple(validation_data) + ('auto',)
        ((len(validation_data) == (3 if self._is_multiclass() else 2))
            or _raise(ValueError(f'len(validation_data) = {len(validation_data)}, but should be {3 if self._is_multiclass() else 2}')))

        patch_size = self.config.train_patch_size
        axes = self.config.axes.replace('C','')
        b = self.config.train_completion_crop if self.config.train_shape_completion else 0
        div_by = self._axes_div_by(axes)
        [(p-2*b) % d == 0 or _raise(ValueError(
            "'train_patch_size' - 2*'train_completion_crop' must be divisible by {d} along axis '{a}'".format(a=a,d=d) if self.config.train_shape_completion else
            "'train_patch_size' must be divisible by {d} along axis '{a}'".format(a=a,d=d)
         )) for p,d,a in zip(patch_size,div_by,axes)]

        if not self._model_prepared:
            self.prepare_for_training()

        data_kwargs = dict (
            n_rays           = self.config.n_rays,
            patch_size       = self.config.train_patch_size,
            grid             = self.config.grid,
            shape_completion = self.config.train_shape_completion,
            b                = self.config.train_completion_crop,
            use_gpu          = self.config.use_gpu,
            foreground_prob  = self.config.train_foreground_only,
            n_classes        = self.config.n_classes,
            sample_ind_cache = self.config.train_sample_cache,
            debug_plots      = debug_plots,
            debug_plot_dir   = debug_plot_dir,
            debug_run_name   = debug_run_name,
        )
        worker_kwargs = dict(workers=workers, use_multiprocessing=workers>1)
        if IS_KERAS_3_PLUS:
            data_kwargs['keras_kwargs'] = worker_kwargs
            fit_kwargs = {}
        else:
            fit_kwargs = worker_kwargs

        # generate validation data and store in numpy arrays
        n_data_val = len(validation_data[0])
        classes_val = self._parse_classes_arg(validation_data[2], n_data_val) if self._is_multiclass() else None
        n_take = self.config.train_n_val_patches if self.config.train_n_val_patches is not None else n_data_val
        _data_val = MyStarDistData2D(validation_data[0],validation_data[1], classes=classes_val, batch_size=n_take, length=1,edt_mode= mode, **data_kwargs)
        data_val = _data_val[2]

        # expose data generator as member for general diagnostics
        self.data_train = MyStarDistData2D(X, Y, classes=classes, batch_size=self.config.train_batch_size,
                                         augmenter=augmenter, length=epochs*steps_per_epoch, **data_kwargs)

        if self.config.train_tensorboard:
            # show dist for three rays
            _n = min(3, self.config.n_rays)
            channel = axes_dict(self.config.axes)['C']
            output_slices = [[slice(None)]*4,[slice(None)]*4]
            output_slices[1][1+channel] = slice(0,(self.config.n_rays//_n)*_n, self.config.n_rays//_n)
            if self._is_multiclass():
                _n = min(3, self.config.n_classes)
                output_slices += [[slice(None)]*4]
                output_slices[2][1+channel] = slice(1,1+(self.config.n_classes//_n)*_n, self.config.n_classes//_n)

            if IS_TF_1:
                for cb in self.callbacks:
                    if isinstance(cb,CARETensorBoard):
                        cb.output_slices = output_slices
                        # target image for dist includes dist_mask and thus has more channels than dist output
                        cb.output_target_shapes = [None,[None]*4,None]
                        cb.output_target_shapes[1][1+channel] = data_val[1][1].shape[1+channel]
            elif self.basedir is not None and not any(isinstance(cb,CARETensorBoardImage) for cb in self.callbacks):
                self.callbacks.append(CARETensorBoardImage(model=self.keras_model, data=data_val, log_dir=str(self.logdir/'logs'/'images'),
                                                           n_images=3, prob_out=False, output_slices=output_slices))

        fit = self.keras_model.fit_generator if (IS_TF_1 and not IS_KERAS_3_PLUS) else self.keras_model.fit
        history = fit(iter(self.data_train), validation_data=data_val,
                      epochs=epochs, steps_per_epoch=steps_per_epoch,
                      **fit_kwargs,
                      callbacks=self.callbacks, verbose=1,
                      # set validation batchsize to training batchsize (only works for tf >= 2.2)
                      **(dict(validation_batch_size = self.config.train_batch_size) if _tf_version_at_least("2.2.0") else {}))
        self._training_finished()

        return history


    def _instances_from_prediction(
        self,
        img_shape,
        prob,
        dist,
        points=None,
        prob_class=None,
        prob_thresh=None,
        nms_thresh=None,
        overlap_label=None,
        return_labels=True,
        scale=None,
        fname=None,
        **nms_kwargs,
    ):
        """
        if points is None     -> dense prediction
        if points is not None -> sparse prediction

        if prob_class is None     -> single class prediction
        if prob_class is not None -> multi class prediction
        """
        if prob_thresh is None:
            prob_thresh = self.thresholds.prob
        if nms_thresh is None:
            nms_thresh = self.thresholds.nms
        if overlap_label is not None:
            raise NotImplementedError("overlap_label not supported for 2D yet!")

        # Keep debug-plot kwargs for merged postprocessing and avoid leaking them
        # into dense-mode non_maximum_suppression.
        merge_debug_kwargs = {
            "debug_plots": nms_kwargs.pop("debug_plots", False),
            "debug_plot_dir": nms_kwargs.pop("debug_plot_dir", None),
            "debug_run_name": nms_kwargs.pop("debug_run_name", None),
            "debug_plot_level": nms_kwargs.pop("debug_plot_level", "basic"),
        }

        stage_outputs = None

        # Sparse prediction: this is your custom merged-polygon path.
        if points is not None:
            postprocess_result = postprocess_instances_merge_polygons(
                dist,
                prob,
                points,
                img_shape,
                nms_thresh=nms_thresh,
                fname=fname,
                **merge_debug_kwargs,
                **nms_kwargs,
            )

            if len(postprocess_result) == 12:
                (
                    points,
                    probi,
                    disti,
                    indsi,
                    new_label,
                    new_coord,
                    new_points,
                    new_prob,
                    final_cell_shapes,
                    final_scores,
                    final_coords,
                    stage_outputs,
                ) = postprocess_result
            else:
                (
                    points,
                    probi,
                    disti,
                    indsi,
                    new_label,
                    new_coord,
                    new_points,
                    new_prob,
                    final_cell_shapes,
                    final_scores,
                    final_coords,
                ) = postprocess_result
                stage_outputs = None

            if prob_class is not None:
                prob_class = prob_class[indsi]

        # Dense prediction: original StarDist NMS path.
        else:
            points, probi, disti, inds, new_label, new_coord, new_points, new_prob, final_cell_shapes = non_maximum_suppression(
                dist,
                prob,
                img_shape,
                grid=self.config.grid,
                prob_thresh=prob_thresh,
                nms_thresh=nms_thresh,
                **nms_kwargs,
            )

            final_scores = new_prob
            final_coords = new_coord

            if prob_class is not None:
                inds = tuple(p // g for p, g in zip(points.T, self.config.grid))
                prob_class = prob_class[inds]

        if scale is not None:
            if not (isinstance(scale, dict) and "X" in scale and "Y" in scale):
                raise ValueError("scale must be a dictionary with entries for 'X' and 'Y'")
            rescale = (1 / scale["Y"], 1 / scale["X"])
            points = points * np.array(rescale).reshape(1, 2)
        else:
            rescale = (1, 1)

        if return_labels:
            labels = new_label
        else:
            labels = None

        coord = new_coord

        res_dict = dict(
            coord=coord,
            points=new_points,
            prob=new_prob,
        )

        if stage_outputs is not None:
            res_dict["stage_outputs"] = stage_outputs

        if prob_class is not None:
            prob_class = np.asarray(prob_class)
            class_id = np.argmax(prob_class, axis=-1)
            res_dict.update(dict(class_prob=prob_class, class_id=class_id))

        return labels, res_dict, final_cell_shapes, final_scores, final_coords

    def predict_instances_with_stage_outputs(self, *args, **kwargs):
        """
        Predict final instances and also return intermediate postprocessing stages.

        This is opt-in and does not change predict_instances(...).
        The returned tuple remains:
            labels, details, final_cell_shapes, final_scores, final_coords

        The intermediate stages are stored in:
            details["stage_outputs"]
        """
        nms_kwargs = kwargs.get("nms_kwargs")
        if nms_kwargs is None:
            nms_kwargs = {}
        else:
            nms_kwargs = dict(nms_kwargs)

        nms_kwargs["return_stage_outputs"] = True
        kwargs["nms_kwargs"] = nms_kwargs

        return self.predict_instances(*args, **kwargs)

    def _axes_div_by(self, query_axes):
        self.config.backbone == 'unet' or _raise(NotImplementedError())
        query_axes = axes_check_and_normalize(query_axes)
        assert len(self.config.unet_pool) == len(self.config.grid)
        div_by = dict(zip(
            self.config.axes.replace('C',''),
            tuple(p**self.config.unet_n_depth * g for p,g in zip(self.config.unet_pool,self.config.grid))
        ))
        return tuple(div_by.get(a,1) for a in query_axes)


    @property
    def _config_class(self):
        return Config2D
    

