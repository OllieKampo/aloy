import aloy.control.controllers as controllers
import aloy.control.pid as pid

if __name__ == "__main__":
    mvc = controllers.MultiVariateController()
    a = pid.PIDController(1.0, 0.0, 0.0)
    b = pid.PIDController(2.0, 0.0, 0.0)
    c = pid.PIDController(2.0, 0.0, 0.0)
    d = pid.PIDController(3.0, 0.0, 0.0)
    e = pid.PIDController(4.0, 0.0, 0.0)
    mvc.add_module(
        "out1",
        {"in1": a, "in2": b},
        mode="sum"
    )
    mvc.add_module(
        "out2",
        {"in3": c},
        mode="sum"
    )
    mvc.add_module(
        "out3",
        {"in4": d, "in5": e},
        mode="sum"
    )
    mvc.cascade_output_to_input("out1", "in3")
    mvc.cascade_output_to_input("out2", "in4")
    mvc.cascade_output_to_input("out1", "in5")
    control_output = mvc.control_output(
        {"in1": 0, "in2": 2, "in3": 3, "in4": 4, "in5": 5},
        {"in1": 1, "in2": 1},
        0.1
    )
    print(control_output)
