module AnovaGLM

using Statistics, StatsBase, LinearAlgebra, Distributions, Reexport, Printf, AnovaBase
@reexport using GLM, AnovaBase
import StatsBase: fit!, fit
import StatsModels: TableRegressionModel, vectorize, ModelFrame, ModelMatrix, response, asgn
import GLM: glm, 
            # Model
            LinPredModel, AbstractGLM, GeneralizedLinearModel, LinearModel, 
            LmResp, GlmResp, 
            # Pred
            LinPred, DensePred, 
            DensePredChol, SparsePredChol, QRCompactWY, SparseMatrixCSC, 
            # prediction
            installbeta!, delbeta!, linpred, linpred!,
            updateμ!, cholfactors, 
            # other
            FP, BlasReal, Link, dispersion, deviance, dof, dof_residual, nobs
using AnovaBase: select_super_interaction, extract_contrasts, canonicalgoodnessoffit, subformula, predictors, dof_asgn, lrt_nested, ftest_nested, _diff, _diffn
import AnovaBase: anova, nestedmodels, anovatable, prednames
export anova_lm, anova_glm

include("anova.jl")
include("fit.jl")
include("io.jl")
end
