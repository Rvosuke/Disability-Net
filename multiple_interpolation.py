import pandas as pd
import miceforest as mf

# Fill in the file path for the missing data set
load_path=''
# Fill in the file path for the dataset after imputation
save_path=''
# Loading missing datasets
data=pd.read_excel(load_path)

kernel=mf.ImputationKernel(
        data,
        datasets=4,
        save_all_iterations=True,
        random_state=1
)
# Run the MICE algorithm for 2 iterations
# Parameters for LightGBM can be passed in the mice() function.
# For more details, see https://pypi.org/project/miceforest/
kernel.mice(2)
print(kernel)

# Return the completed dataset.
completed_dataset=kernel.complete_data(dataset=2)
completed_dataset.to_excel(save_path,index=False)