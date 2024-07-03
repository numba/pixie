from numba.core import sigutils, extending, config
import inspect


# Adapted from: https://github.com/numba/numba/blob/7f056946c7a69f8739c07ef5a1bdb4b4b5be72cd/numba/core/decorators.py  # noqa: E501


def aot(signature_or_function=None, locals={}, cache=False,
        pipeline_class=None, boundscheck=None, **options):

    forceobj = options.get('forceobj', False)
    if forceobj:
        raise ValueError("object mode is not supported")
    npm = options.get('nopython', None)
    if npm is False:
        raise ValueError("nopython mode must be True if supplied")

    if "_target" in options:
        # Set the "target_backend" option if "_target" is defined.
        options['target_backend'] = options['_target']
    target = options.pop('_target', 'cpu')

    options['nopython'] = True
    options['_nrt'] = False  # NO NRT AVAILABLE AT PRESENT
    # Set these to get unoptimised output with debug info present.
    # options['debug'] = True
    # options['_dbg_optnone'] = True
    options['forceinline'] = True
    options['boundscheck'] = boundscheck
    options['no_cpython_wrapper'] = True

    # Handle signature
    if signature_or_function is None:
        # No signature, no function
        pyfunc = None
        sigs = None
    elif isinstance(signature_or_function, list):
        # A list of signatures is passed
        pyfunc = None
        sigs = signature_or_function
    elif sigutils.is_signature(signature_or_function):
        # A single signature is passed
        pyfunc = None
        sigs = [signature_or_function]
    else:
        # A function is passed
        pyfunc = signature_or_function
        sigs = None

    dispatcher_args = {}
    if pipeline_class is not None:
        dispatcher_args['pipeline_class'] = pipeline_class
    wrapper = _jit(sigs, locals=locals, target=target, cache=cache,
                   targetoptions=options, **dispatcher_args)
    if pyfunc is not None:
        return wrapper(pyfunc)
    else:
        return wrapper


class LazyDispatcher():

    def __init__(self, sigs, dispatcher):
        self._sigs = sigs
        self._dispatcher = dispatcher

    def __call__(self, *args, **kwargs):
        self._dispatcher(*args, **kwargs)


def _jit(sigs, locals, target, cache, targetoptions, **dispatcher_args):

    from numba.core.target_extension import resolve_dispatcher_from_str
    dispatcher = resolve_dispatcher_from_str(target)

    def wrapper(func):
        if extending.is_jitted(func):
            raise TypeError(
                "A jit decorator was called on an already jitted function "
                f"{func}.  If trying to access the original python "
                f"function, use the {func}.py_func attribute."
            )

        if not inspect.isfunction(func):
            raise TypeError(
                "The decorated object is not a function (got type "
                f"{type(func)})."
            )

        if config.ENABLE_CUDASIM and target == 'cuda':
            from numba import cuda
            return cuda.jit(func)
        if config.DISABLE_JIT and not target == 'npyufunc':
            return func

        disp = dispatcher(py_func=func, locals=locals,
                          targetoptions=targetoptions,
                          **dispatcher_args)

        return LazyDispatcher(sigs, disp)

    return wrapper
