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
    fn new(start: Vec<f64>, goal: Vec<f64>) -> Self {
        Tree {
            nodes: vec![start, goal],
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
    fn new(start: Vec<f64>, goal: Vec<f64>) -> (Self, Tree) {
        (CostedTree { costs: vec![0.0] }, Tree::new(start, goal))
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
pub fn simple_rrt(
    _py: Python,
    start: Vec<f64>,
    goal: Vec<f64>,
    boundaries: Vec<f64>,
    obstacles: Vec<Vec<f64>>,
    max_iterations: usize,
) -> PyResult<Tree> {
    return rrt(_py, start, goal, boundaries, obstacles, max_iterations, -1.0);
}

#[pyfunction]
/// Runs the RRT algorithm.
/// 
/// # Arguments
/// 
/// * `start` - A vector of floats representing the start point.
/// * `goal` - A vector of floats representing the goal point.
/// * `boundaries` - A vector of floats representing the boundaries of the space.
/// * `obstacles` - A vector of vectors representing the obstacles.
/// * `max_iterations` - An integer representing the maximum number of iterations to run the
/// algorithm for.
/// * `step_size` - A float representing the maximum distance the random point can be from the
/// nearest neighbor. If this is negative, the random point will be generated within the boundaries
/// and not moved towards the nearest neighbor.
/// 
/// # Returns
/// 
/// A vector of vectors representing the tree.
/// 
/// # Examples
/// 
/// ```rust
/// use rrt::planning::rrt;
/// 
/// let start = vec![0.0, 0.0];
/// let goal = vec![10.0, 10.0];
/// let boundaries = vec![10.0, 10.0];
/// let obstacles = vec![
///    vec![5.0, 5.0],
///    vec![5.0, 6.0],
///    vec![6.0, 5.0],
///    vec![6.0, 6.0],
/// ];
/// let max_iterations = 1000;
/// let step_size = 1.0;
/// let tree = rrt(start, goal, boundaries, obstacles, max_iterations, step_size);
/// ```
/// 
/// ```python
/// from aloy.rost.planning.rrt import rrt
///
/// start = [0.0, 0.0]
/// goal = [10.0, 10.0]
/// rrt(start, goal)
/// ```
pub fn rrt(
    _py: Python,
    start: Vec<f64>,
    goal: Vec<f64>,
    boundaries: Vec<f64>,
    obstacles: Vec<Vec<f64>>,
    max_iterations: usize,
    step_size: f64,
) -> PyResult<Tree> {
    let mut tree = Tree::new(start, goal);
    let mut i = 0;
    while i < max_iterations {
        // Generate random point.
        let (random_point, nearest_index) = new_random_point(
            &tree, &obstacles, &boundaries, step_size);
        // Move the nearest neighbor towards the random point.
        let new_point = move_point(&tree[nearest_index], &random_point, step_size);
        // Add new point to the tree.
        tree.push(new_point);
        // Connect the nearest neighbor to the new point.
        tree.connect(nearest_index, tree.len()-1);
        // Stop if the nearest neighbor was the goal.
        if nearest_index == 1 {
            return Ok(tree);
        }
        i += 1;
    }
    return Ok(tree);
}


/// Generates a random point within the given boundaries.
/// 
/// # Arguments
/// 
/// * `tree` - A vector of vectors representing the tree.
/// * `obstacles` - A vector of vectors representing the obstacles.
/// * `boundaries` - A vector of floats representing the boundaries of the space.
/// * `step_size` - A float representing the maximum distance the random point can be from the
/// nearest neighbor.
/// 
/// # Returns
/// 
/// A vector of floats representing the random point.
/// 
fn new_random_point(
    tree: &Tree,
    obstacles: &Vec<Vec<f64>>,
    boundaries: &Vec<f64>,
    step_size: f64
) -> (Vec<f64>, usize) {
    if step_size < 0.0 {
        loop {
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
            return (random_point, nearest_index);
        }
    }
    loop {
        // Generate random point as a float between 0 and the boundary.
        let random_point = vec![
            rand::thread_rng().gen_range(0.0..boundaries[0]),
            rand::thread_rng().gen_range(0.0..boundaries[1]),
        ];
        // Find nearest neighbor to the random point.
        let nearest_index = nearest_neighbor(&tree, &random_point);
        // Move the random point closer to the nearest neighbor.
        let moved_point = move_point(&tree[nearest_index], &random_point, step_size);
        // Check if random point is not in an obstacle.
        if !collision_free(&moved_point, &obstacles) {
            continue;
        }
        return (moved_point, nearest_index);
    }
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


/// Moves a point towards a random point within a given step size.
///
/// # Arguments
///
/// * `nearest_point` - The nearest point to the random point.
/// * `random_point` - The random point to move towards.
/// * `step_size` - The maximum distance to move towards the random point.
///
/// # Returns
///
/// The new point after moving towards the random point.
fn move_point(nearest_point: &Vec<f64>, random_point: &Vec<f64>, step_size: f64) -> Vec<f64> {
    let distance = distance(&nearest_point, &random_point);
    if distance <= step_size {
        return random_point.clone();
    }
    let theta = (random_point[1] - nearest_point[1]).atan2(random_point[0] - nearest_point[0]);
    let mut new_point = nearest_point.clone();
    new_point[0] += step_size * theta.cos();
    new_point[1] += step_size * theta.sin();
    return new_point;
}


/// Checks if a given point is collision-free with a set of obstacles.
///
/// # Arguments
///
/// * `point` - The coordinates of the point to check for collision.
/// * `obstacles` - A list of obstacles, where each obstacle is represented by a list of coordinates.
///
/// # Returns
///
/// Returns `true` if the point is collision-free, `false` otherwise.
fn collision_free(point: &Vec<f64>, obstacles: &Vec<Vec<f64>>) -> bool {
    for obstacle in obstacles {
        if distance(&point, &obstacle) < 1.0 {
            return false;
        }
    }
    return true;
}


/// Calculates the Euclidean distance between two points in n-dimensional space.
///
/// # Arguments
///
/// * `point_a` - The coordinates of the first point.
/// * `point_b` - The coordinates of the second point.
///
/// # Returns
///
/// The Euclidean distance between `point_a` and `point_b`.
fn distance(point_a: &Vec<f64>, point_b: &Vec<f64>) -> f64 {
    let mut sum = 0.0;
    for (a, b) in point_a.iter().zip(point_b.iter()) {
        sum += (a - b).powi(2);
    }
    return sum.sqrt();
}
