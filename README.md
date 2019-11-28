single task score

|            | Accuracy | Precision | Recall |  F1   |  AUC  |
| :--------: | :------: | :-------: | :----: | :---: | :---: |
| aggression |  0.924   |   0.788   | 0.785  | 0.787 | 0.870 |
|   attack   |  0.937   |   0.825   | 0.784  | 0.803 | 0.875 |
|  toxicity  |  0.937   |   0.811   | 0.838  | 0.824 | 0.898 |



multi task score

|            | Accuracy | Precision | Recall |  F1   |  AUC  |
| :--------: | :------: | :-------: | :----: | :---: | :---: |
| aggression |  0.932   |   0.823   | 0.789  | 0.805 | 0.876 |
|   attack   |  0.939   |   0.804   | 0.830  | 0.817 | 0.895 |
|  toxicity  |  0.942   |   0.845   | 0.822  | 0.833 | 0.895 |

aggression+attack

|            | Accuracy | Precision | Recall |  F1   |  AUC  |
| :--------: | :------: | :-------: | :----: | :---: | :---: |
| aggression |  0.929   |   0.802   | 0.800  | 0.801 | 0.879 |
|   attack   |  0.935   |   0.783   | 0.842  | 0.811 | 0.898 |

aggression+toxicity

|            | Accuracy | Precision | Recall |  F1   |  AUC  |
| :--------: | :------: | :-------: | :----: | :---: | :---: |
| aggression |  0.931   |   0.813   | 0.797  | 0.805 | 0.879 |
|  toxicity  |  0.943   |   0.840   | 0.834  | 0.837 | 0.900 |

attack+toxicity

|          | Accuracy | Precision | Recall |  F1   |  AUC  |
| :------: | :------: | :-------: | :----: | :---: | :---: |
|  attack  |  0.940   |   0.816   | 0.820  | 0.818 | 0.892 |
| toxicity |  0.944   |   0.861   | 0.815  | 0.837 | 0.893 |

