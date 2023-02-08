#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <thread>
#include <chrono>
#include <iostream>

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

PYBIND11_MODULE(example, m) {
    m.doc() = "pybind11 example plugin";
    m.def("add", &add, "A function that adds two numbers");
    // m.def("first_where", &first_where<float>, "A function that finds the first element where a condition is true");
    m.def("arg_first_where", &arg_first_where, "A function that finds the first index where a condition is true");

    py::class_<Mult>(m, "Mult")
        .def(py::init<float>())
        .def("multiply", &Mult::multiply)
        .def("multiply_vector", &Mult::multiply_vector)
        .def("multiply_tuple", &Mult::multiply_tuple)
        // .def("function_that_takes_a_lambda", &Mult::function_that_takes_a_lambda)
        .def("function_concurrent_callable", &Mult::function_concurrent_callable)
        .def_readwrite("multiplier", &Mult::multiplier);
}