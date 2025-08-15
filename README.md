# GraIL-ART
## Inductive relation prediction experiments

All train-graph and ind-test-graph pairs of graphs can be found in the `data` folder. We use WN18RR_v1 as a runninng example for illustrating the steps.

### GraIL-ART
To start training a GraIL-ART model.
In the GraIL-ART directory:
a. Generate subgraph datasets and corresponding ID mapping files:
   ```
   python roughsets/subgraph_generator.py --dataset WN18RR_v1 --output_dir ./roughsets/output 
   ```
b. Learn rules:
   ```
   python roughsets/phase1_rule_learning.py --dataset WN18RR_v1 --confidence_threshold 0.6 --support_threshold 3 --verbose  
   ```

c. Training:
   ```
   python roughsets/phase2_joint_training.py --dataset fb237_v1 --experiment_name WN18RR_v1_hop2 --gpu 0 --num_epochs 40 --aux_loss_weight 0.3 --rule_dropout 0.1 --enclosing_sub_graph
   ```

d. Test performance:
   ```
   python roughsets/run_inductive_evaluation.py \
     --dataset WN18RR_v1_ind \
     --model_dataset WN18RR_v1 \
     --experiment_name WN18RR_v1_hop2 \
     --hop 2 \
     --enclosing_sub_graph \
     --output_dir roughsets/output \
   ```
The trained model and the logs are stored in `experiments` folder. Note that to ensure a fair comparison, we test all models on the same negative triplets. In order to do that in the current setup, we store the sampled negative triplets while evaluating GraIL and use these later to evaluate other baseline models.


## Transductive experiments

The full transductive datasets used in these experiments are present in the `data` folder.

# GraIL-ART
# GraIL-ART
# GraIL-ART
