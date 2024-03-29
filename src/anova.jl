# ===========================================================================================
# Main API
"""
    anova(<glmmodels>...; test::Type{<: GoodnessOfFit},  <keyword arguments>)
    anova(<anovamodel>; test::Type{<: GoodnessOfFit},  <keyword arguments>)
    anova(test::Type{<: GoodnessOfFit}, <glmmodels>...;  <keyword arguments>)
    anova(test::Type{<: GoodnessOfFit}, <anovamodel>;  <keyword arguments>)

Analysis of variance.

Return `AnovaResult{M, test, N}`. See [`AnovaResult`](@ref) for details.

# Arguments
* `glmmodels`: model objects
    1. `TableRegressionModel{<: LinearModel}` fitted by `GLM.lm`
    2. `TableRegressionModel{<: GeneralizedLinearModel}` fitted by `GLM.glm`
    If mutiple models are provided, they should be nested and the last one is the most complex.
* `anovamodel`: wrapped model objects; `FullModel` and `NestedModels`.
* `test`: test statistics for goodness of fit. Available tests are [`LikelihoodRatioTest`](@ref) ([`LRT`](@ref)) and [`FTest`](@ref). The default is based on the model type.
    1. `TableRegressionModel{<: LinearModel}`: `FTest`.
    2. `TableRegressionModel{<: GeneralizedLinearModel}`: based on distribution function, see `canonicalgoodnessoffit`.

# Other keyword arguments
* When one model is provided:  
    1. `type` specifies type of anova (1, 2 or 3). Default value is 1.
* When multiple models are provided:  
    1. `check`: allows to check if models are nested. Defalut value is true. Some checkers are not implemented now.

!!! note
    For fitting new models and conducting anova at the same time, see [`anova_lm`](@ref) for `LinearModel`, [`anova_glm`](@ref) for `GeneralizedLinearModel`.
"""
anova(::Type{<: GoodnessOfFit}, ::Vararg{TableRegressionModel{<: Union{LinearModel, GeneralizedLinearModel}}})

anova(models::Vararg{TableRegressionModel{<: LinearModel}}; 
        test::Type{<: GoodnessOfFit} = FTest,
        kwargs...) = 
    anova(test, models...; kwargs...)

anova(models::FullModel{<: TableRegressionModel{<: LinearModel}}; 
        test::Type{<: GoodnessOfFit} = FTest,
        kwargs...) = 
    anova(test, models; kwargs...)

anova(aovm::NestedModels{<: TableRegressionModel{<: LinearModel}}; 
        test::Type{<: GoodnessOfFit} = FTest,
        kwargs...) = 
    anova(test, aovm; kwargs...)

anova(models::Vararg{TableRegressionModel{<: GeneralizedLinearModel}}; 
        test::Type{<: GoodnessOfFit} = canonicalgoodnessoffit(models[1].model.rr.d),
        kwargs...) = 
    anova(test, models...; kwargs...)

anova(aovm::FullModel{<: TableRegressionModel{<: GeneralizedLinearModel}}; 
        test::Type{<: GoodnessOfFit} = canonicalgoodnessoffit(aovm.model.model.rr.d),
        kwargs...) = 
    anova(test, aovm; kwargs...)

anova(aovm::NestedModels{<: TableRegressionModel{<: GeneralizedLinearModel}}; 
        test::Type{<: GoodnessOfFit} = canonicalgoodnessoffit(aovm.model[1].model.rr.d),
        kwargs...) = 
    anova(test, aovm; kwargs...)

# ==================================================================================================================
# ANOVA by F test 
# LinearModels
const TRM_LM = TableRegressionModel{<: Union{LinearModel, GeneralizedLinearModel{<: GLM.GlmResp{T, <: Normal, IdentityLink}}}} where T

anova(::Type{FTest}, 
    trm::TableRegressionModel{<: Union{LinearModel, GeneralizedLinearModel}}; 
    type::Int = 1, kwargs...) = anova(FTest, FullModel(trm, type, isnullable(trm.model), true); kwargs...)

function anova(::Type{FTest}, aovm::FullModel{<: TRM_LM})
    assign = asgn(predictors(aovm))
    fullpred = predictors(aovm.model)
    fullasgn = asgn(fullpred)
    df = tuple(dof_asgn(assign)...)
    varβ = vcov(aovm.model.model)
    β = aovm.model.model.pp.beta0
    offset = first(assign) + last(fullasgn) - last(assign) - 1
    if aovm.type == 1
        fs = abs2.(cholesky(Hermitian(inv(varβ))).U * β) 
        fstat = ntuple(last(fullasgn) - offset) do fix
            sum(fs[findall(==(fix + offset), fullasgn)]) / df[fix]
        end
    elseif aovm.type == 2
        fstat = ntuple(last(fullasgn) - offset) do fix
            s1 = sort!(collect(select_super_interaction(fullpred, fix + offset)))
            s2 = setdiff(s1, fix + offset)
            select1 = findall(in(s1), fullasgn)
            select2 = findall(in(s2), fullasgn)
            (β[select1]' * (varβ[select1, select1] \ β[select1]) - β[select2]' * (varβ[select2, select2] \ β[select2])) / df[fix]
        end
    else
        # calculate block by block
        fstat = ntuple(last(fullasgn) - offset) do fix
            select = findall(==(fix + offset), fullasgn)
            β[select]' * (varβ[select, select] \ β[select]) / df[fix]
        end
    end
    σ² = dispersion(aovm.model.model, true)
    devs = @. fstat * σ² * df
    dfr = round(Int, dof_residual(aovm.model))
    pvalue = @. ccdf(FDist(df, dfr), abs(fstat))
    AnovaResult(aovm, FTest, df, devs, fstat, pvalue, NamedTuple())
end

function anova(::Type{FTest}, 
    aovm::FullModel{<: TableRegressionModel{<: Union{LinearModel, GeneralizedLinearModel}}}; kwargs...)
    devs = deviances(aovm; kwargs...)
    assign = asgn(collect(predictors(aovm)))
    #length(vdf) ≡ length(devs) + 1 && popfirst!(vdf)
    df = tuple(dof_asgn(assign)...)
    msr = devs ./ df
    fstat = msr ./ dispersion(aovm.model.model, true)
    dfr = round(Int, dof_residual(aovm.model))
    pvalue = @. ccdf(FDist(df, dfr), abs(fstat))
    AnovaResult(aovm, FTest, df, devs, fstat, pvalue, NamedTuple())
end

# ----------------------------------------------------------------------------------------
# ANOVA for genaralized linear models
# λ = -2ln(𝓛(̂θ₀)/𝓛(θ)) ~ χ²ₙ , n = difference of predictors
anova(::Type{LRT}, 
        trm::TableRegressionModel{<: Union{LinearModel, GeneralizedLinearModel}}; 
        type::Int = 1, kwargs...) = anova(LRT, FullModel(trm, type, isnullable(trm.model), true); kwargs...)  

function anova(::Type{LRT}, 
        aovm::FullModel{<: TableRegressionModel{<: Union{LinearModel, GeneralizedLinearModel}}})
    Δdev = deviances(aovm)
    assign = asgn(collect(predictors(aovm)))
    #isnullable(trm.model) || popfirst!(vdf)
    df = tuple(dof_asgn(assign)...)
    # den = last(ss) / (nobs(trm) - dof(trm) + 1)
    # lrstat = ss[1:end - 1] ./ den
    σ² = dispersion(aovm.model.model, true)
    lrstat = Δdev ./ σ²
    n = length(lrstat)
    dev = push!(collect(Δdev), deviance(aovm.model))
    i = n
    while i > 0
        dev[i] += dev[i + 1]
        i -= 1
    end
    pval = @. ccdf(Chisq(df), abs(lrstat))
    AnovaResult(aovm, LRT, df, tuple(dev[2:end]...), lrstat, pval, NamedTuple())
end

# =================================================================================================================
# Nested models 

function anova(::Type{FTest}, 
        trms::Vararg{M}; 
        check::Bool = true) where {M <: TableRegressionModel{<: Union{LinearModel, GeneralizedLinearModel}}}  
    df = dof.(trms)
    ord = sortperm(collect(df))
    df = df[ord]
    trms = trms[ord]
    dfr = round.(Int, dof_residual.(trms))
    # May exist some floating point error from dof_residual
    # check comparable and nested
    check && @warn "Could not check whether models are nested: results may not be meaningful"
    ftest_nested(NestedModels(trms), df, dfr, deviance.(trms), dispersion(last(trms).model, true))
end

function anova(::Type{LRT}, 
        trms::Vararg{M}; 
        check::Bool = true) where {M <: TableRegressionModel{<: Union{LinearModel, GeneralizedLinearModel}}}  
    df = dof.(trms)
    ord = sortperm(collect(df))
    trms = trms[ord]
    df = df[ord]
    # check comparable and nested
    check && @warn "Could not check whether models are nested: results may not be meaningful"
    lrt_nested(NestedModels(trms), df, deviance.(trms), dispersion(last(trms).model, true))
end

anova(::Type{FTest}, aovm::NestedModels{M}) where {M <: TableRegressionModel{<: Union{LinearModel, GeneralizedLinearModel}}} =
    ftest_nested(aovm, dof.(aovm.model), round.(Int, dof_residual.(aovm.model)), deviance.(aovm.model), dispersion(last(aovm.model).model, true))

anova(::Type{LRT}, aovm::NestedModels{M}) where {M <: TableRegressionModel{<: Union{LinearModel, GeneralizedLinearModel}}} =
    lrt_nested(aovm, dof.(aovm.model), deviance.(aovm.model), dispersion(last(aovm.model).model, true))
# =================================================================================================================================
# Fit new models

"""
    anova_lm(X, y; test::Type{<: GoodnessOfFit} = FTest, <keyword arguments>) 

    anova_lm(test::Type{<: GoodnessOfFit}, X, y; <keyword arguments>)

    anova(test::Type{<: GoodnessOfFit}, ::Type{LinearModel}, X, y; 
        type::Int = 1, 
        <keyword arguments>)

ANOVA for simple linear regression.

# Arguments
* `X` and `y` can be a `Matrix` and a `Vector` or a `Formula` and a `Tables.jl` compatible data. 
* `test`: test statistics for goodness of fit.

# Keyword arguments
* `test`: test statistics for goodness of fit.
* `type` specifies type of anova (1, 2 or 3). Default value is 1.
* `dropcollinear` controls whether or not `lm` accepts a model matrix which is less-than-full rank. If true (the default), only the first of each set of linearly-dependent columns is used. The coefficient for redundant linearly dependent columns is 0.0 and all associated statistics are set to NaN.

`anova_lm` generate a `TableRegressionModel` object, which is fitted by `lm`.
"""
anova_lm(X, y; 
        test::Type{<: GoodnessOfFit} = FTest, 
        kwargs...) = 
    anova(test, LinearModel, X, y; kwargs...)

anova_lm(test::Type{<: GoodnessOfFit}, X, y; kwargs...) = 
    anova(test, LinearModel, X, y; kwargs...)

function anova(test::Type{<: GoodnessOfFit}, ::Type{LinearModel}, X, y; 
        type::Int = 1, 
        kwargs...)
    model = lm(X, y; kwargs...)
    anova(test, model; type)
end

"""
    anova_glm(X, y, d::UnivariateDistribution, l::Link = canonicallink(d); 
            test::Type{<: GoodnessOfFit} = canonicalgoodnessoffit(d), <keyword arguments>)

    anova_glm(test::Type{<: GoodnessOfFit}, X, y, d::UnivariateDistribution, l::Link = canonicallink(d); <keyword arguments>)

    anova(test::Type{<: GoodnessOfFit}, X, y, d::UnivariateDistribution, l::Link = canonicallink(d); <keyword arguments>)

ANOVA for genaralized linear models.

# Arguments
* `X` and `y` can be a `Matrix` and a `Vector` or a `Formula` and a `Tables.jl` compatible data.
* `d`: a `GLM.UnivariateDistribution`.
* `l`: a `GLM.Link`
* `test`: test statistics for goodness of fit based on distribution function. See `canonicalgoodnessoffit`.

For other keyword arguments, see `fit`.
"""
anova_glm(X, y, 
        d::UnivariateDistribution, l::Link = canonicallink(d); 
        test::Type{<: GoodnessOfFit} = canonicalgoodnessoffit(d), 
        kwargs...) = 
    anova(test, GeneralizedLinearModel, X, y, d, l; kwargs...)

anova_glm(test::Type{<: GoodnessOfFit}, X, y, 
        d::UnivariateDistribution, l::Link = canonicallink(d); 
        kwargs...) = 
    anova(test, GeneralizedLinearModel, X, y, d, l; kwargs...)

function anova(test::Type{<: GoodnessOfFit}, ::Type{GeneralizedLinearModel}, X, y, 
        d::UnivariateDistribution, l::Link = canonicallink(d);
        type::Int = 1,
        kwargs...)
    trm = glm(X, y, d, l; kwargs...)
    anova(test, trm; type, kwargs... )
end