use pyo3::prelude::*;
use rand::Rng;

#[pyclass(module="rrt", subclass)]
pub struct Tree {
    nodes: Vec<Vec<f64>>,
    adjacency_list: Vec<Vec<usize>>,
}

#[pymethods]
impl Tree {
    #[new]
    fn new(start: Vec<f64>) -> Self {
        Tree {
            nodes: vec![start],
            adjacency_list: vec![vec![]],
        }
    }

    #[getter]
    fn nodes(&self) -> Vec<Vec<f64>> {
        self.nodes.clone()
    }

    #[getter]
    fn adjacency_list(&self) -> Vec<Vec<usize>> {
        self.adjacency_list.clone()
    }

    fn push(&mut self, node: Vec<f64>) {
        self.nodes.push(node);
        self.adjacency_list.push(vec![]);
    }

    fn connect(&mut self, parent_index: usize, child_index: usize) {
        self.adjacency_list[parent_index].push(child_index);
    }

    fn __getitem__(&self, index: usize) -> Vec<f64> {
        self.nodes[index].clone()
    }

    fn __len__(&self) -> usize {
        self.nodes.len()
    }
}

impl Tree {
    pub fn len(&self) -> usize {
        self.nodes.len()
    }
}

impl std::ops::Index<usize> for Tree {
    type Output = Vec<f64>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.nodes[index]
    }
}

#[pyclass(extends=Tree)]
pub struct CostedTree {
    costs: Vec<f64>,
}

#[pymethods]
impl CostedTree {
    #[new]
    fn new(start: Vec<f64>) -> (Self, Tree) {
        (CostedTree { costs: vec![0.0] }, Tree::new(start))
    }

    fn costs(&self) -> Vec<f64> {
        self.costs.clone()
    }

    fn push(mut self_: PyRefMut<'_, Self>, node: Vec<f64>) {
        self_.costs.push(0.0);
        let mut super_ = self_.into_super();
        super_.push(node);
    }
}


#[pyfunction]
pub fn rapidly_exploring_random_tree(
    _py: Python,
    start: Vec<f64>,
    goal: Vec<f64>,
    boundaries: Vec<f64>,
    obstacles: Vec<Vec<f64>>,
    step_size: f64,
    max_iterations: usize,
) -> PyResult<Tree> {
    let mut tree = Tree::new(start);
    let mut i = 0;
    while i < max_iterations {
        // Generate random point as a float between 0 and the boundary.
        let random_point = vec![
            rand::thread_rng().gen_range(0.0..boundaries[0]),
            rand::thread_rng().gen_range(0.0..boundaries[1]),
        ];
        // Check if random point is not in an obstacle.
        if !collision_free(&random_point, &obstacles) {
            continue;
        }
        // Find nearest neighbor to the random point.
        let nearest_index = nearest_neighbor(&tree, &random_point);
        // Add new point to the tree.
        tree.push(random_point);
        // 
        tree.connect(nearest_index, tree.len()-1);
        if distance(&tree[tree.len()-1], &goal) < step_size {
            tree.push(goal);
            tree.connect(tree.len()-2, tree.len()-1);
            return Ok(tree);
        }
        i += 1;
    }
    return Ok(tree);
}


/// Finds the nearest neighbor in a tree to a given point.
///
/// # Arguments
///
/// * `tree` - A vector of vectors representing the tree.
/// * `point` - A vector representing the point to find the nearest neighbor to.
///
/// # Returns
///
/// A vector representing the nearest neighbor to the given point.
fn nearest_neighbor(tree: &Tree, point: &Vec<f64>) -> usize {
    let mut nearest_index: usize = 0;
    let nearest_point = &tree[nearest_index];
    let mut min_distance = distance(&nearest_point, &point);
    for index in 1..tree.len() {
        let node = &tree[index];
        let distance = distance(&node, &point);
        if distance < min_distance {
            min_distance = distance;
            nearest_index = index;
        }
    }
    return nearest_index;
}


// fn new_state(nearest_point: &Vec<f64>, random_point: &Vec<f64>, step_size: f64) -> Vec<f64> {
//     let mut new_point = nearest_point.clone();
//     let distance = distance(&nearest_point, &random_point);
//     if distance <= step_size {
//         return random_point.clone();
//     }
//     let theta = (random_point[1] - nearest_point[1]).atan2(random_point[0] - nearest_point[0]);
//     new_point[0] += step_size * theta.cos();
//     new_point[1] += step_size * theta.sin();
//     return new_point;
// }


fn collision_free(point: &Vec<f64>, obstacles: &Vec<Vec<f64>>) -> bool {
    for obstacle in obstacles {
        if distance(&point, &obstacle) < 1.0 {
            return false;
        }
    }
    return true;
}


fn distance(point_a: &Vec<f64>, point_b: &Vec<f64>) -> f64 {
    let mut sum = 0.0;
    for (a, b) in point_a.iter().zip(point_b.iter()) {
        sum += (a - b).powi(2);
    }
    return sum.sqrt();
}
