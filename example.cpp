#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <thread>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <deque>
#include <vector>
#include <cmath>

namespace py = pybind11;

int add(int i, int j) {
    return i + j;
}

int arg_first_where(py::array_t<float> array, std::function<bool(float)> f) {
    auto r = array.unchecked<1>();
    for (int i = 0; i < r.shape(0); i++) {
        if (f(r(i))) {
            return i;
        }
    }
    return -1;
}

// template <typename T>
// T first_where(py::array_t<T> array, std::function<bool(T)> f, py::object default_ = py::none()) {
//     // Return the first element where the function f returns true.
//     for (auto i : array) {
//         if (f(i)) {
//             return i;
//         }
//     }
//     return default_;
// }

class Mult {
    public:
        float multiplier;

        Mult(float multiplier) : multiplier(multiplier) {}

        float multiply(float input) const {
            return multiplier * input;
        }

        std::vector<float> multiply_vector(std::vector<float> input) const {
            std::vector<float> output = {};
            for (auto i : input) {
                output.push_back(this->multiply(i));
            }
            return output;
        }

        py::tuple multiply_tuple(py::tuple input) const {
            py::tuple output = py::tuple(input.size());
            for (int i = 0; i < input.size(); i++) {
                output[i] = this->multiply(input[i].cast<float>());
            }
            return output;
        }

        // void function_that_takes_a_lambda(std::function<void(float)> f) {
        //     f(1.0);
        // }

        void function_concurrent_callable() const {
            py::gil_scoped_release release;
            std::cout << "Sleeping for 2 seconds" << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(2));
            std::cout << "Woke up" << std::endl;
            py::gil_scoped_acquire acquire;

            auto list = py::list();
            list.append(1);
            list.append(2);
            list.append(3);
        }
};

class ControlledSystem {
public:
    float get_setpoint();
    float get_control_input();
    void set_control_output(float control_output, float delta_time);
};

class Controller {
public:
    float control_output(float control_input, float setpoint, float delta_time, float abs_tol);
    float latest_error;
};

std::vector<int> get_turning_points(std::vector<float> error_values) {
    std::vector<int> turning_points = {};
    if (error_values.size() < 2) {
        return turning_points;
    }
    float prev_error = error_values[0];
    for (int i = 1; i < error_values.size(); i++) {
        float error = error_values[i];
        if (error * prev_error < 0.0) {
            turning_points.push_back(i);
        }
        prev_error = error;
    }
    return turning_points;
}

float simulate_control(py::object& controlled_system_,
                       py::object& controller_,
                       int ticks,
                       float delta_time,
                       bool penalise_oscillation = true,
                       bool penalise_overshoot = true,
                       int lead_ticks = 1,
                       int lag_ticks = 1) {
    
    py::object ControlledSystem_ = py::module_::import("control.controllers").attr("ControlledSystem");
    // if (!py::isinstance<ControlledSystem_>(controlled_system_)) {
    //     throw std::invalid_argument("Controlled system must be an instance of ControlledSystem.");
    // }
    py::object Controller_ = py::module_::import("control.controllers").attr("Controller");
    // if (!py::isinstance<Controller_>(controller_)) {
    //     throw std::invalid_argument("Controller must be an instance of Controller.");
    // }

    ControlledSystem* controlled_system = controlled_system_.cast<ControlledSystem*>();
    Controller* controller = controller_.cast<Controller*>();

    if (ticks < 1)
        throw std::invalid_argument("Ticks must be greater than or equal to 1.");
    if (delta_time <= 0.0)
        throw std::invalid_argument("Delta time must be greater than 0.");
    if (lead_ticks < 1)
        throw std::invalid_argument("Lead ticks must be greater than or equal to 1.");
    if (lag_ticks < 1)
        throw std::invalid_argument("Lag ticks must be greater than or equal to 1.");

    float setpoint = controlled_system->get_setpoint();
    std::vector<float> error_values(ticks);
    std::vector<float> control_outputs(ticks);
    std::vector<float> time_points(ticks);
    for (int i = 0; i < ticks; i++) {
        time_points[i] = i * delta_time;
    }

    float control_output = 0.0;
    if (lead_ticks == 1 && lag_ticks == 1) {
        for (int tick = 0; tick < ticks; tick++) {
            float control_input = controlled_system->get_control_input();
            control_output = controller->control_output(control_input, setpoint, delta_time, 0.0);
            controlled_system->set_control_output(control_output, delta_time);
            error_values[tick] = controller->latest_error;
            control_outputs[tick] = control_output;
        }
    } else {
        std::deque<float> input_queue;
        std::deque<float> output_queue;
        for (int tick = 0; tick < ticks; tick++) {
            float control_input = controlled_system->get_control_input();
            input_queue.push_back(control_input);
            if (input_queue.size() == lead_ticks) {
                control_input = input_queue.front();
                input_queue.pop_front();
                control_output = controller->control_output(control_input, setpoint, delta_time, 0.0);
                output_queue.push_back(control_output);
                if (output_queue.size() == lag_ticks) {
                    control_output = output_queue.front();
                    output_queue.pop_front();
                    controlled_system->set_control_output(control_output, delta_time);
                }
            }
            error_values[tick] = controller->latest_error;
            control_outputs[tick] = control_output;
        }
    }

    // Calculate integral of absolute error over time.
    std::vector<float> itae(ticks);
    for (int i = 0; i < ticks; i++)
        itae[i] = std::abs(error_values[i]) * delta_time;
    
    // Penalise oscillation by multiplying the error between turning points,
    // where later turning points are penalised more than earlier ones.
    if (penalise_oscillation) {
        std::vector<int> points = get_turning_points(error_values);
        if (!points.empty()) {
            for (int i = 0; i < points.size() - 1; ++i) {
                int start = points[i];
                int end = points[i + 1];
                for (int j = start; j < end; ++j) {
                    itae[j] *= ((points.size() + 1) - i);
                }
            }
            int last_point = points.back();
            for (int j = last_point; j < ticks; ++j) {
                itae[j] *= 2.0;
            }
        }
    }

    // Penalise overshoot by doubling the overshooting error values.
    if (penalise_overshoot) {
        if (error_values[0] < 0.0) {
            for (int i = 0; i < ticks; i++) {
                if (error_values[i] > 0.0) {
                    itae[i] *= 2.0;
                }
            }
        } else {
            for (int i = 0; i < ticks; i++) {
                if (error_values[i] < 0.0) {
                    itae[i] *= 2.0;
                }
            }
        }
    }

    return std::accumulate(itae.begin(), itae.end(), 0.0);
}

PYBIND11_MODULE(example, m) {
    m.doc() = "pybind11 example plugin";
    m.def("add", &add, "A function that adds two numbers");
    // m.def("first_where", &first_where<float>, "A function that finds the first element where a condition is true");
    m.def("arg_first_where", &arg_first_where, "A function that finds the first index where a condition is true");

    m.def("simulate_control", &simulate_control, "Simulate a control system with a controller.");

    py::class_<Mult>(m, "Mult")
        .def(py::init<float>())
        .def("multiply", &Mult::multiply)
        .def("multiply_vector", &Mult::multiply_vector)
        .def("multiply_tuple", &Mult::multiply_tuple)
        // .def("function_that_takes_a_lambda", &Mult::function_that_takes_a_lambda)
        .def("function_concurrent_callable", &Mult::function_concurrent_callable)
        .def_readwrite("multiplier", &Mult::multiplier);
}