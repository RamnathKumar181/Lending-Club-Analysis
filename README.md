# Lending-Club-Analysis

We create a model using the gradient boosting algorithm to cut down on the noise and improve performance. Our goal of this project is to draw light on the importance on the "verification_status" field, which usually plays a big role in determining whether a given individual is sanctioned the loan or not. We disprove this common practice using attribution methods built on top of the gradient boosting algorithm. For more details about our experiments and results, please refer to [Paper](Is_verification_status_important?.pdf)!

## Data
We build on top of peer-to-peer lending dataset known as the LendingClubDataset which is publicly available.

## Requirements
- Python 3.6
- sklearn : <code> pip install sklearn </code>
- numpy : <code> pip install numpy </code>

## System Used
1. CPU: Intel(R) Xeon(R) CPU @ 2.30GHz
2. GPU: 1xTesla K80 , having 2496 CUDA cores, compute 3.7,  12GB(11.439GB Usable) GDDR5  VRAM
3. RAM: 12.6GB

## Acknowledgements
I would like to thank Dr. Hussain Yaganti for giving me the opportunity to work on this project
