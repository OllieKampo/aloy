MultiVariateSystemController takes MultivVariateController as argument and extracts its inputs and outputs automatically.
Can an AutoSystemController then also take a MutliVariateController as argument?

MultiVariateController can then have
add_controller(..., out_var_name: str)
cascade(out_var_name: str, in_var_name: str)
This removes the output var as a control output, and instead cascades it as the setpoint to the input var, therefore also removing the need to give a setpoint for the input var.
