module AnovaGLM

using Statistics, StatsBase, LinearAlgebra, Distributions, Reexport, Printf, AnovaBase
@reexport using GLM, AnovaBase
import StatsBase: fit!, fit
using StatsModels: TableRegressionModel, vectorize, ModelFrame, ModelMatrix, response, asgn
using GLM: 
    LmResp, GlmResp, 
    DensePredChol, LinPred, DensePred, SparsePredChol, 
    installbeta!, updateμ!, cholfactors, 
    FP, BlasReal
import GLM: delbeta!, linpred, linpred!, deviance
    # Pred: QRCompactWY, SparseMatrixCSC, 
using AnovaBase: 
    select_super_interaction, extract_contrasts, 
    canonicalgoodnessoffit, FixDispDist, 
    subformula, predictors, dof_asgn, dof_aovres, deviance, 
    lrt_nested, ftest_nested, _diff, _diffn
import AnovaBase: anova, nestedmodels, anovatable, prednames, dof_aov
export anova_lm, anova_glm

include("anova.jl")
include("fit.jl")
include("io.jl")
end
