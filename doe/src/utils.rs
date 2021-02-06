use ndarray::{s, Array1, Array2, ArrayBase, Data, Ix2, NdFloat};
use ndarray_stats::DeviationExt;
use num_traits::Signed;

pub fn pdist<F: NdFloat + Signed>(x: &ArrayBase<impl Data<Elem = F>, Ix2>) -> Array1<F> {
    let nrows = x.nrows();
    let size: usize = (nrows - 1) * nrows / 2;
    let mut res: Array1<F> = Array1::zeros(size);
    let mut k = 0;
    for i in 0..nrows {
        for j in (i + 1)..nrows {
            let a = x.slice(s![i, ..]);
            let b = x.slice(s![j, ..]);
            res[k] = F::from(a.l2_dist(&b).unwrap()).unwrap();
            k += 1;
        }
    }
    res
}

pub fn cdist<F: NdFloat + Signed>(
    xa: &ArrayBase<impl Data<Elem = F>, Ix2>,
    xb: &ArrayBase<impl Data<Elem = F>, Ix2>,
) -> Array2<F> {
    let ma = xa.nrows();
    let mb = xb.nrows();
    let na = xa.ncols();
    let nb = xb.ncols();
    if na != nb {
        panic!(
            "cdist: operands should have same nb of columns. Found {} and {}",
            na, nb
        );
    }
    let mut res = Array2::zeros((ma, mb));
    for i in 0..ma {
        for j in 0..mb {
            let a = xa.slice(s![i, ..]);
            let b = xb.slice(s![j, ..]);
            res[[i, j]] = F::from(a.l2_dist(&b).unwrap()).unwrap();
        }
    }

    res
}
