# Excess Capacity Learning (ECL) vs Complementary Learning Systems (CLS)

1. Minimal ECL model:
   - Number of parameters
   - Number of training epochs 
   - Degree of explicit regularization (M: potentially not necessary since the implicit regularization is being introduced in excess capacity?)
2. Components of the CLS model: 
   - Hippocampus - sufficient capacity 
   - Neocortex - constrained capacity
   - Interactions between H and N (e.g., replay)
   - Temporal learning dynamics (memorization in hippocampus → generalization in neocortex)
3. Qualitative way to assess the properties:
   - Particular experiments from the original paper https://miashs-www.u-ga.fr/prevert/MasterIC2A/SpecialiteSC/FichiersPDF/Why%20there%20are%20complementary%20learning%20systems%20in%20the%20hippocampus%20and%20neocortex%20insights%20from%20th.pdf
   - The qualitative ways to assess the general properties below:
     - Temporal: individual instances are memorized first and generalizable patterns are acquired later
     - No/very little catastrophic interference
     - Encoding individual experiences
     - Learning generalizable patterns
