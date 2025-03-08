# This file is automatically generated by pyo3_stub_gen
# ruff: noqa: E501, F401

import numpy
import numpy.typing
import typing
from enum import Enum, auto

class CorrelationSpec:
    ...

class Egor:
    r"""
    Optimizer constructor
    
       fun: array[n, nx]) -> array[n, ny]
            the function to be minimized
            fun(x) = [obj(x), cstr_1(x), ... cstr_k(x)] where
               obj is the objective function [n, nx] -> [n, 1]
               cstr_i is the ith constraint function [n, nx] -> [n, 1]
               an k the number of constraints (n_cstr)
               hence ny = 1 (obj) + k (cstrs)
            cstr functions are expected be negative (<=0) at the optimum.
            This constraints will be approximated using surrogates, so
            if constraints are cheap to evaluate better to pass them through run(fcstrs=[...])
   
        n_cstr (int):
            the number of constraints which will be approximated by surrogates (see `fun` argument)
   
        cstr_tol (list(n_cstr + n_fcstr,)):
            List of tolerances for constraints to be satisfied (cstr < tol),
            list size should be equal to n_cstr + n_fctrs where n_cstr is the `n_cstr` argument
            and `n_fcstr` the number of constraints passed as functions.
    
        xspecs (list(XSpec)) where XSpec(xtype=FLOAT|INT|ORD|ENUM, xlimits=[<f(xtype)>] or tags=[strings]):
            Specifications of the nx components of the input x (eg. len(xspecs) == nx)
            Depending on the x type we get the following for xlimits:
            * when FLOAT: xlimits is [float lower_bound, float upper_bound],
            * when INT: xlimits is [int lower_bound, int upper_bound],
            * when ORD: xlimits is [float_1, float_2, ..., float_n],
            * when ENUM: xlimits is just the int size of the enumeration otherwise a list of tags is specified
              (eg xlimits=[3] or tags=["red", "green", "blue"], tags are there for documention purpose but
               tags specific values themselves are not used only indices in the enum are used hence
               we can just specify the size of the enum, xlimits=[3]),
    
        n_start (int > 0):
            Number of runs of infill strategy optimizations (best result taken)
    
        n_doe (int >= 0):
            Number of samples of initial LHS sampling (used when DOE not provided by the user).
            When 0 a number of points is computed automatically regarding the number of input variables
            of the function under optimization.
    
        doe (array[ns, nt]):
            Initial DOE containing ns samples:
                either nt = nx then only x are specified and ns evals are done to get y doe values,
                or nt = nx + ny then x = doe[:, :nx] and y = doe[:, nx:] are specified  
    
        regr_spec (RegressionSpec flags, an int in [1, 7]):
            Specification of regression models used in gaussian processes.
            Can be RegressionSpec.CONSTANT (1), RegressionSpec.LINEAR (2), RegressionSpec.QUADRATIC (4) or
            any bit-wise union of these values (e.g. RegressionSpec.CONSTANT | RegressionSpec.LINEAR)
    
        corr_spec (CorrelationSpec flags, an int in [1, 15]):
            Specification of correlation models used in gaussian processes.
            Can be CorrelationSpec.SQUARED_EXPONENTIAL (1), CorrelationSpec.ABSOLUTE_EXPONENTIAL (2),
            CorrelationSpec.MATERN32 (4), CorrelationSpec.MATERN52 (8) or
            any bit-wise union of these values (e.g. CorrelationSpec.MATERN32 | CorrelationSpec.MATERN52)
    
        infill_strategy (InfillStrategy enum)
            Infill criteria to decide best next promising point.
            Can be either InfillStrategy.EI, InfillStrategy.WB2 or InfillStrategy.WB2S.

        cstr_strategy (ConstraintStrategy enum)
            Constraint management either use the mean value or upper bound
            Can be either ConstraintStrategy.MV (default) or ConstraintStrategy.UTB.
    
        q_points (int > 0):
            Number of points to be evaluated to allow parallel evaluation of the function under optimization.
    
        par_infill_strategy (ParInfillStrategy enum)
            Parallel infill criteria (aka qEI) to get virtual next promising points in order to allow
            q parallel evaluations of the function under optimization (only used when q_points > 1)
            Can be either ParInfillStrategy.KB (Kriging Believer),
            ParInfillStrategy.KBLB (KB Lower Bound), ParInfillStrategy.KBUB (KB Upper Bound),
            ParInfillStrategy.CLMIN (Constant Liar Minimum)
    
        infill_optimizer (InfillOptimizer enum)
            Internal optimizer used to optimize infill criteria.
            Can be either InfillOptimizer.COBYLA or InfillOptimizer.SLSQP
    
        kpls_dim (0 < int < nx)
            Number of components to be used when PLS projection is used (a.k.a KPLS method).
            This is used to address high-dimensional problems typically when nx > 9.
    
        trego (bool)
            When true, TREGO algorithm is used, otherwise classic EGO algorithm is used.
    
        n_clusters (int)
            Number of clusters used by the mixture of surrogate experts (default is 1).
            When set to 0, the number of cluster is determined automatically and refreshed every
            10-points addition (should say 'tentative addition' because addition may fail for some points
            but it is counted anyway).
            When set to negative number -n, the number of clusters is determined automatically in [1, n]
            this is used to limit the number of trials hence the execution time.
      
        n_optmod (int >= 1)
            Number of iterations between two surrogate models training (hypermarameters optimization)
            otherwise previous hyperparameters are re-used. The default value is 1 meaning surrogates are
            properly trained at each iteration. The value is used as a modulo of iteration number. For instance,
            with a value of 3, after the first iteration surrogate are trained at iteration 3, 6, 9, etc.  
    
        target (float)
            Known optimum used as stopping criterion.
    
        outdir (String)
            Directory to write optimization history and used as search path for warm start doe
    
        warm_start (bool)
            Start by loading initial doe from <outdir> directory
    
        hot_start (int >= 0 or None)
            When hot_start>=0 saves optimizer state at each iteration and starts from a previous checkpoint
            if any for the given hot_start number of iterations beyond the max_iters nb of iterations.
            In an unstable environment were there can be crashes it allows to restart the optimization
            from the last iteration till stopping criterion is reached. Just use hot_start=0 in this case.
            When specifying an extended nb of iterations (hot_start > 0) it can allow to continue till max_iters +
            hot_start nb of iters is reached (provided the stopping criterion is max_iters)
            Checkpoint information is stored in .checkpoint/egor.arg binary file.
    
        seed (int >= 0)
            Random generator seed to allow computation reproducibility.
    """
    def __new__(cls,xspecs,n_cstr = ...,cstr_tol = ...,n_start = ...,n_doe = ...,doe = ...,regr_spec = ...,corr_spec = ...,infill_strategy = ...,q_points = ...,par_infill_strategy = ...,infill_optimizer = ...,kpls_dim = ...,trego = ...,n_clusters = ...,n_optmod = ...,target = ...,outdir = ...,warm_start = ...,hot_start = ...,seed = ...): ...
    def minimize(self, fun,max_iters = ..., fcstrs = ...) -> OptimResult:
        r"""
        This function finds the minimum of a given function `fun`
        
        # Parameters
            max_iters:
                the iteration budget, number of fun calls is n_doe + q_points * max_iters.

            fcstrs: 
                list of constraints functions defined as g(x, return_grad): (ndarray[nx], bool) -> float or ndarray[nx,]
                If the given `return_grad` boolean is `False` the function has to return the constraint float value
                to be made negative by the optimizer (which drives the input array `x`).
                Otherwise the function has to return the gradient (ndarray[nx,]) of the constraint funtion
                wrt the `nx` components of `x`.
        
        # Returns
            optimization result
                x_opt (array[1, nx]): x value where fun is at its minimum subject to constraints
                y_opt (array[1, nx]): fun(x_opt)
        """
        ...

    def suggest(self, x_doe,y_doe) -> numpy.typing.NDArray[numpy.float64]:
        r"""
        This function gives the next best location where to evaluate the function
        under optimization wrt to previous evaluations.
        The function returns several point when multi point qEI strategy is used.
        
        # Parameters
            x_doe (array[ns, nx]): ns samples where function has been evaluated
            y_doe (array[ns, 1 + n_cstr]): ns values of objecctive and constraints
            
        
        # Returns
            (array[1, nx]): suggested location where to evaluate objective and constraints
        """
        ...

    def get_result_index(self, y_doe) -> int:
        r"""
        This function gives the best evaluation index given the outputs
        of the function (objective wrt constraints) under minimization.
        
        # Parameters
            y_doe (array[ns, 1 + n_cstr]): ns values of objective and constraints
            
        # Returns
            index in y_doe of the best evaluation
        """
        ...

    def get_result(self, x_doe,y_doe) -> OptimResult:
        r"""
        This function gives the best result given inputs and outputs
        of the function (objective wrt constraints) under minimization.
        
        # Parameters
            x_doe (array[ns, nx]): ns samples where function has been evaluated
            y_doe (array[ns, 1 + n_cstr]): ns values of objective and constraints
            
        # Returns
            optimization result
                x_opt (array[1, nx]): x value where fun is at its minimum subject to constraints
                y_opt (array[1, nx]): fun(x_opt)
        """
        ...


class ExpectedOptimum:
    val: float
    tol: float

class GpMix:
    r"""
    Gaussian processes mixture builder
    
        n_clusters (int >= 0)
            Number of clusters used by the mixture of surrogate experts.
            When set to 0, the number of cluster is determined automatically and refreshed every
            10-points addition (should say 'tentative addition' because addition may fail for some points
            but failures are counted anyway).
    
        regr_spec (RegressionSpec flags, an int in [1, 7]):
            Specification of regression models used in mixture.
            Can be RegressionSpec.CONSTANT (1), RegressionSpec.LINEAR (2), RegressionSpec.QUADRATIC (4) or
            any bit-wise union of these values (e.g. RegressionSpec.CONSTANT | RegressionSpec.LINEAR)
    
        corr_spec (CorrelationSpec flags, an int in [1, 15]):
            Specification of correlation models used in mixture.
            Can be CorrelationSpec.SQUARED_EXPONENTIAL (1), CorrelationSpec.ABSOLUTE_EXPONENTIAL (2),
            CorrelationSpec.MATERN32 (4), CorrelationSpec.MATERN52 (8) or
            any bit-wise union of these values (e.g. CorrelationSpec.MATERN32 | CorrelationSpec.MATERN52)
    
        recombination (Recombination.Smooth or Recombination.Hard (default))
            Specify how the various experts predictions are recombined
            * Smooth: prediction is a combination of experts prediction wrt their responsabilities,
            the heaviside factor which controls steepness of the change between experts regions is optimized
            to get best mixture quality.
            * Hard: prediction is taken from the expert with highest responsability
            resulting in a model with discontinuities.
    
        theta_init ([nx] where nx is the dimension of inputs x)
            Initial guess for GP theta hyperparameters.
            When None the default is 1e-2 for all components
    
        theta_bounds ([[lower_1, upper_1], ..., [lower_nx, upper_nx]] where nx is the dimension of inputs x)
            Space search when optimizing theta GP hyperparameters
            When None the default is [1e-6, 1e2] for all components
    
        kpls_dim (0 < int < nx where nx is the dimension of inputs x)
            Number of components to be used when PLS projection is used (a.k.a KPLS method).
            This is used to address high-dimensional problems typically when nx > 9.
    
        n_start (int >= 0)
            Number of internal GP hyperpameters optimization restart (multistart)
    
        seed (int >= 0)
            Random generator seed to allow computation reproducibility.
    """
    def __new__(cls,n_clusters = ...,regr_spec = ...,corr_spec = ...,recombination = ...,theta_init = ...,theta_bounds = ...,kpls_dim = ...,n_start = ...,seed = ...): ...
    def fit(self, xt:numpy.typing.NDArray[numpy.float64], yt:numpy.typing.NDArray[numpy.float64]) -> Gpx:
        r"""
        Fit the parameters of the model using the training dataset to build a trained model
        
        Parameters
            xt (array[nsamples, nx]): input samples
            yt (array[nsamples, 1]): output samples
        
        Returns Gpx object
            the fitted Gaussian process mixture  
        """
        ...


class Gpx:
    r"""
    A trained Gaussian processes mixture
    """
    @staticmethod
    def builder(n_clusters = ...,regr_spec = ...,corr_spec = ...,recombination = ...,theta_init = ...,theta_bounds = ...,kpls_dim = ...,n_start = ...,seed = ...) -> GpMix:
        r"""
        Get Gaussian processes mixture builder aka `GpMix`
        
        See `GpMix` constructor
        """
        ...

    def __repr__(self) -> str:
        r"""
        Returns the String representation from serde json serializer
        """
        ...

    def __str__(self) -> str:
        r"""
        Returns a String informal representation
        """
        ...

    def save(self, filename:str) -> bool:
        r"""
        Save Gaussian processes mixture in a file.
        If the filename has .json JSON human readable format is used
        otherwise an optimized binary format is used.
        
        Parameters
            filename with .json or .bin extension (string)
                file generated in the current directory
        
        Returns True if save succeeds otherwise False
        """
        ...

    @staticmethod
    def load(filename:str) -> Gpx:
        r"""
        Load Gaussian processes mixture from file.
        
        Parameters
            filename (string)
                json filepath generated by saving a trained Gaussian processes mixture
        """
        ...

    def predict(self, x:numpy.typing.NDArray[numpy.float64]) -> numpy.typing.NDArray[numpy.float64]:
        r"""
        Predict output values at nsamples points.
        
        Parameters
            x (array[nsamples, nx])
                input values
        
        Returns
            the output values at nsamples x points (array[nsamples, 1])
        """
        ...

    def predict_var(self, x:numpy.typing.NDArray[numpy.float64]) -> numpy.typing.NDArray[numpy.float64]:
        r"""
        Predict variances at nsample points.
        
        Parameters
            x (array[nsamples, nx])
                input values
        
        Returns
            the variances of the output values at nsamples input points (array[nsamples, 1])
        """
        ...

    def predict_gradients(self, x:numpy.typing.NDArray[numpy.float64]) -> numpy.typing.NDArray[numpy.float64]:
        r"""
        Predict surrogate output derivatives at nsamples points.
        
        Parameters
            x (array[nsamples, nx])
                input values
        
        Returns
            the output derivatives at nsamples x points (array[nsamples, nx]) wrt inputs
            The ith column is the partial derivative value wrt to the ith component of x at the given samples.
        """
        ...

    def predict_var_gradients(self, x:numpy.typing.NDArray[numpy.float64]) -> numpy.typing.NDArray[numpy.float64]:
        r"""
        Predict variance derivatives at nsamples points.
        
        Parameters
            x (array[nsamples, nx])
                input values
        
        Returns
            the variance derivatives at nsamples x points (array[nsamples, nx]) wrt inputs
            The ith column is the partial derivative value wrt to the ith component of x at the given samples.
        """
        ...

    def sample(self, x:numpy.typing.NDArray[numpy.float64], n_traj:int) -> numpy.typing.NDArray[numpy.float64]:
        r"""
        Sample gaussian process trajectories.
        
        Parameters
            x (array[nsamples, nx])
                locations of the sampled trajectories
            n_traj number of trajectories to generate
        
        Returns
            the trajectories as an array[nsamples, n_traj]
        """
        ...

    def dims(self) -> tuple[int, int]:
        r"""
        Get the input and output dimensions of the surrogate
        
        Returns
            the couple (nx, ny)
        """
        ...

    def training_data(self) -> tuple[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64]]:
        r"""
        Get the nt training data points used to fit the surrogate
        
        Returns
            the couple (ndarray[nt, nx], ndarray[nt,])
        """
        ...

    def thetas(self) -> numpy.typing.NDArray[numpy.float64]:
        r"""
        Get optimized thetas hyperparameters (ie once GP experts are fitted)
        
        Returns
            thetas as an array[n_clusters, nx or kpls_dim]
        """
        ...

    def variances(self) -> numpy.typing.NDArray[numpy.float64]:
        r"""
        Get GP expert variance (ie posterior GP variance)
        
        Returns
            variances as an array[n_clusters]
        """
        ...

    def likelihoods(self) -> numpy.typing.NDArray[numpy.float64]:
        r"""
        Get reduced likelihood values gotten when fitting the GP experts
        
        Maybe used to compare various parameterization
        
        Returns
            likelihood as an array[n_clusters]
        """
        ...


class OptimResult:
    x_opt: numpy.typing.NDArray[numpy.float64]
    y_opt: numpy.typing.NDArray[numpy.float64]
    x_doe: numpy.typing.NDArray[numpy.float64]
    y_doe: numpy.typing.NDArray[numpy.float64]

class RegressionSpec:
    ...

class SparseGpMix:
    r"""
    Sparse Gaussian processes mixture builder
    
        n_clusters (int)
            Number of clusters used by the mixture of surrogate experts (default is 1).
            When set to 0, the number of cluster is determined automatically and refreshed every
            10-points addition (should say 'tentative addition' because addition may fail for some points
            but it is counted anyway).
            When set to negative number -n, the number of clusters is determined automatically in [1, n]
            this is used to limit the number of trials hence the execution time.
    
        corr_spec (CorrelationSpec flags, an int in [1, 15]):
            Specification of correlation models used in mixture.
            Can be CorrelationSpec.SQUARED_EXPONENTIAL (1), CorrelationSpec.ABSOLUTE_EXPONENTIAL (2),
            CorrelationSpec.MATERN32 (4), CorrelationSpec.MATERN52 (8) or
            any bit-wise union of these values (e.g. CorrelationSpec.MATERN32 | CorrelationSpec.MATERN52)
    
        recombination (Recombination.Smooth or Recombination.Hard)
            Specify how the various experts predictions are recombined
            * Smooth: prediction is a combination of experts prediction wrt their responsabilities,
            the heaviside factor which controls steepness of the change between experts regions is optimized
            to get best mixture quality.
            * Hard: prediction is taken from the expert with highest responsability
            resulting in a model with discontinuities.
    
        kpls_dim (0 < int < nx where nx is the dimension of inputs x)
            Number of components to be used when PLS projection is used (a.k.a KPLS method).
            This is used to address high-dimensional problems typically when nx > 9.
    
        n_start (int >= 0)
            Number of internal GP hyperpameters optimization restart (multistart)
    
        method (SparseMethod.FITC or SparseMethod.VFE)
            Sparse method to be used (default is FITC)
    
        seed (int >= 0)
            Random generator seed to allow computation reproducibility.
    """
    def __new__(cls,corr_spec = ...,theta_init = ...,theta_bounds = ...,kpls_dim = ...,n_start = ...,nz = ...,z = ...,method = ...,seed = ...): ...
    def fit(self, xt:numpy.typing.NDArray[numpy.float64], yt:numpy.typing.NDArray[numpy.float64]) -> SparseGpx:
        r"""
        Fit the parameters of the model using the training dataset to build a trained model
        
        Parameters
            xt (array[nsamples, nx]): input samples
            yt (array[nsamples, 1]): output samples
        
        Returns Sgp object
            the fitted Gaussian process mixture  
        """
        ...


class SparseGpx:
    r"""
    A trained Gaussian processes mixture
    """
    @staticmethod
    def builder(corr_spec = ...,theta_init = ...,theta_bounds = ...,kpls_dim = ...,n_start = ...,nz = ...,z = ...,method = ...,seed = ...) -> SparseGpMix:
        r"""
        Get Gaussian processes mixture builder aka `GpSparse`
        
        See `GpSparse` constructor
        """
        ...

    def __repr__(self) -> str:
        r"""
        Returns the String representation from serde json serializer
        """
        ...

    def __str__(self) -> str:
        r"""
        Returns a String informal representation
        """
        ...

    def save(self, filename:str) -> bool:
        r"""
        Save Gaussian processes mixture in a file.
        If the filename has .json JSON human readable format is used
        otherwise an optimized binary format is used.
        
        Parameters
            filename with .json or .bin extension (string)
                file generated in the current directory
        
        Returns True if save succeeds otherwise False
        """
        ...

    @staticmethod
    def load(filename:str) -> SparseGpx:
        r"""
        Load Gaussian processes mixture from a json file.
        
        Parameters
            filename (string)
                json filepath generated by saving a trained Gaussian processes mixture
        """
        ...

    def predict(self, x:numpy.typing.NDArray[numpy.float64]) -> numpy.typing.NDArray[numpy.float64]:
        r"""
        Predict output values at nsamples points.
        
        Parameters
            x (array[nsamples, nx])
                input values
        
        Returns
            the output values at nsamples x points (array[nsamples])
        """
        ...

    def predict_var(self, x:numpy.typing.NDArray[numpy.float64]) -> numpy.typing.NDArray[numpy.float64]:
        r"""
        Predict variances at nsample points.
        
        Parameters
            x (array[nsamples, nx])
                input values
        
        Returns
            the variances of the output values at nsamples input points (array[nsamples, 1])
        """
        ...

    def predict_gradients(self, x:numpy.typing.NDArray[numpy.float64]) -> numpy.typing.NDArray[numpy.float64]:
        r"""
        Predict surrogate output derivatives at nsamples points.
        
        Implementation note: central finite difference technique
        on `predict()` function is used which may be subject to numerical issues
        
        Parameters
            x (array[nsamples, nx])
                input values
        
        Returns
            the output derivatives at nsamples x points (array[nsamples, nx]) wrt inputs
            The ith column is the partial derivative value wrt to the ith component of x at the given samples.
        """
        ...

    def predict_var_gradients(self, x:numpy.typing.NDArray[numpy.float64]) -> numpy.typing.NDArray[numpy.float64]:
        r"""
        Predict variance derivatives at nsamples points.
        
        Implementation note: central finite difference technique
        on `predict_var()` function is used which may be subject to numerical issues
        
        Parameters
            x (array[nsamples, nx])
                input values
        
        Returns
            the variance derivatives at nsamples x points (array[nsamples, nx]) wrt inputs
            The ith column is the partial derivative value wrt to the ith component of x at the given samples.
        """
        ...

    def sample(self, x:numpy.typing.NDArray[numpy.float64], n_traj:int) -> numpy.typing.NDArray[numpy.float64]:
        r"""
        Sample gaussian process trajectories.
        
        Parameters
            x (array[nsamples, nx])
                locations of the sampled trajectories
            n_traj number of trajectories to generate
        
        Returns
            the trajectories as an array[nsamples, n_traj]
        """
        ...

    def thetas(self) -> numpy.typing.NDArray[numpy.float64]:
        r"""
        Get optimized thetas hyperparameters (ie once GP experts are fitted)
        
        Returns
            thetas as an array[n_clusters, nx or kpls_dim]
        """
        ...

    def variances(self) -> numpy.typing.NDArray[numpy.float64]:
        r"""
        Get GP expert variance (ie posterior GP variance)
        
        Returns
            variances as an array[n_clusters]
        """
        ...

    def likelihoods(self) -> numpy.typing.NDArray[numpy.float64]:
        r"""
        Get reduced likelihood values gotten when fitting the GP experts
        
        Maybe used to compare various parameterization
        
        Returns
            likelihood as an array[n_clusters]
        """
        ...


class XSpec:
    xtype: XType
    xlimits: list[float]
    tags: list[str]
    def __new__(cls,xtype,xlimits = ...,tags = ...): ...

class InfillOptimizer(Enum):
    Cobyla = auto()
    Slsqp = auto()

class InfillStrategy(Enum):
    Ei = auto()
    Wb2 = auto()
    Wb2s = auto()

class ConstraintStrategy(Enum):
    Mv = auto()
    Utb = auto()

class ParInfillStrategy(Enum):
    Kb = auto()
    Kblb = auto()
    Kbub = auto()
    Clmin = auto()

class Recombination(Enum):
    Hard = auto()
    Smooth = auto()

class Sampling(Enum):
    Lhs = auto()
    FullFactorial = auto()
    Random = auto()
    LhsClassic = auto()
    LhsCentered = auto()
    LhsMaximin = auto()
    LhsCenteredMaximin = auto()

class SparseMethod(Enum):
    Fitc = auto()
    Vfe = auto()

class XType(Enum):
    Float = auto()
    Int = auto()
    Ord = auto()
    Enum = auto()

def lhs(xspecs,n_samples,seed = ...) -> numpy.typing.NDArray[numpy.float64]:
    r"""
    Samples generation using optimized Latin Hypercube Sampling
    
    # Parameters
        xspecs: list of XSpec
        n_samples: number of samples
        seed: random seed
    
    # Returns
       ndarray of shape (n_samples, n_variables)
    """
    ...

def sampling(method,xspecs,n_samples,seed = ...) -> numpy.typing.NDArray[numpy.float64]:
    r"""
    Samples generation using given method
    
    # Parameters
        method: LHS, FULL_FACTORIAL or RANDOM
        xspecs: list of XSpec
        n_samples: number of samples
        seed: random seed
    
    # Returns
       ndarray of shape (n_samples, n_variables)
    """
    ...

def to_specs(xlimits:typing.Sequence[typing.Sequence[float]]) -> typing.Any:
    r"""
    Utility function converting `xlimits` float data list specifying bounds of x components
    to x specified as a list of XType.Float types [egobox.XType]
    
    # Parameters
        xlimits : nx-size list of [lower_bound, upper_bound] where `nx` is the dimension of x
    
    # Returns
        xtypes: nx-size list of XSpec(XType(FLOAT), [lower_bound, upper_bounds]) where `nx` is the dimension of x
    """
    ...

