import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_residual_parameter_correlations(file_path):
    df = pd.read_csv(file_path, skiprows=4)
    objective = df["objective_function"]
    x_filter_col = [col for col in df if col.startswith('x(')]
    x = df[x_filter_col]
    res_filter_col = [col for col in df if col.startswith('F(')]
    residuals = df[res_filter_col]
    
    
    plt.subplot(1, 2, 1)
    objective.plot(logy=True)
    plt.xlabel("Optimization Iteration")
    plt.ylabel("Objective Value")
    plt.subplot(1, 2, 2)
    x.corrwith(objective).plot()
    plt.title('Input Correlation with Objective')
    plt.xlabel('Input Index')
    plt.ylabel('Correlation Coefficient')
    plt.tight_layout()

    plt.figure()
    df.corrwith(residuals).plot()
    plt.title('Correlation Between Residuals and Parameters')
    plt.xlabel('Residual Index')
    plt.ylabel('Correlation Coefficient')
    plt.show()

if __name__ == "__main__":
    import sys
    plot_residual_parameter_correlations(sys.argv[1])