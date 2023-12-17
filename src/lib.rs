use pyo3::prelude::*;
use pyo3::{wrap_pyfunction, wrap_pymodule};


mod planning;


#[pyfunction]
fn vector_add(
    py: Python,
    vector_a: Vec<f64>,
    vector_b: PyObject,
) -> PyResult<Vec<f64>> {
    if vector_a.len() == 0 {
        return Ok(Vec::new());
    }
    let vector_b = if let Ok(b) = vector_b.extract::<f64>(py) {
        vec![b; vector_a.len()]
    } else {
        vector_b.extract::<Vec<f64>>(py)?
    };
    let result = vector_a
        .iter()
        .zip(vector_b.iter())
        .map(|(&a, &b)| a + b)
        .collect();
    return Ok(result);
}


#[pymodule]
#[pyo3(name="vectors")]
fn vectors(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(vector_add, m)?)?;
    Ok(())
}


#[pymodule]
#[pyo3(name="moremath")]
fn moremath_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pymodule!(vectors))?;
    Ok(())
}


#[pymodule]
#[pyo3(name="rrt")]
fn rrt(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(planning::rrt::rrt, m)?)?;
    Ok(())
}


#[pymodule]
#[pyo3(name="planning")]
fn planning_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pymodule!(rrt))?;
    Ok(())
}


#[pymodule]
#[pyo3(name="rost")]
fn rost(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pymodule!(moremath_py))?;
    m.add_wrapped(wrap_pymodule!(planning_py))?;
    Ok(())
}
