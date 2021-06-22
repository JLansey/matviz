import matplotlib.scale as mscale
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import matplotlib.ticker as ticker
import numpy as np

# code to make axis scaled by the cube root
# https://stackoverflow.com/questions/42277989/square-root-scale-using-matplotlib-python

class CubeRootScale(mscale.ScaleBase):
    """
    ScaleBase class for generating square root scale.

    Use:
    from matviz import cbrt_scale
    ax.set_yscale('cuberoot')

    """

    name = 'cuberoot'

    def __init__(self, axis, **kwargs):
        # note in older versions of matplotlib (<3.1), this worked fine.
        # mscale.ScaleBase.__init__(self)

        # In newer versions (>=3.1), you also need to pass in `axis` as an arg
        mscale.ScaleBase.__init__(self, axis)

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(ticker.AutoLocator())
        axis.set_major_formatter(ticker.ScalarFormatter())
        axis.set_minor_locator(ticker.NullLocator())
        axis.set_minor_formatter(ticker.NullFormatter())

    def limit_range_for_scale(self, vmin, vmax, minpos):
        return max(0., vmin), vmax

    class CubeRootTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def transform_non_affine(self, a):
            return np.cbrt(a)

        def inverted(self):
            return CubeRootScale.InvertedCubeRootTransform()

    class InvertedCubeRootTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def transform(self, a):
            return np.array(a) ** 2

        def inverted(self):
            return CubeRootScale.CubeRootTransform()

    def get_transform(self):
        return self.CubeRootTransform()



mscale.register_scale(CubeRootScale)


