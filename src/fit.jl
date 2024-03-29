# ==========================================================================================================
# Backend funcion

function deviances(aovm::FullModel{<: TableRegressionModel{<: Union{LinearModel, GeneralizedLinearModel}}}; kwargs...)
    isa(aovm.model.model.pp, DensePredChol) || throw(
            ArgumentError("Methods for other PredChol types is not implemented; use model with DensesPredChol instead."))
    trm = aovm.model
    if aovm.type ≡ 1
        ids = union(first(aovm.pred_id) - 1, aovm.pred_id)
        devs = -diff(map(i -> deviance(trm, @view(ids[i + 1:end]); kwargs...), eachindex(ids)))
    elseif aovm.type ≡ 2
        devs = zeros(Float64, length(aovm.pred_id))
        # cache fitted
        dict_devs = Dict{Set{Int}, Float64}()
        f = formula(aovm.model).rhs
        @inbounds for (id, del) in enumerate(aovm.pred_id)
            delcoef = select_super_interaction(f, del)
            dev1 = get!(dict_devs, delcoef, deviance(trm, delcoef; kwargs...))
            delete!(delcoef, del)
            dev2 = get!(dict_devs, delcoef, deviance(trm, delcoef; kwargs...))
            devs[id] = dev1 - dev2
        end
    else
        devr = deviance(trm, 0; kwargs...)
        devs = map(del -> deviance(trm, del; kwargs...) - devr, aovm.pred_id)
    end
    # every method end with deviance(trm, 0; kwargs...), ie full model fit.
    deviance(trm, 0; kwargs...)
    installbeta!(trm.model.pp) # ensure model unchanged
    tuple(devs...)
end

submm(trm::TableRegressionModel, exclude) = 
    trm.mm.m[:, map(!in(exclude), trm.mm.assign)]
submm(trm::TableRegressionModel, exclude::Int) = 
    trm.mm.m[:, map(!=(exclude), trm.mm.assign)]

# Backend for LinearModel
# Only used by type 2 SS
function deviance(trm::TableRegressionModel{<: LinearModel}, exclude)
    p = trm.model.pp

    # reschema to calculate new X, reset beta
    X = submm(trm, exclude)
    p.beta0 = p.delbeta = repeat([0], size(X, 2))
    p.scratchbeta = similar(p.beta0)

    # cholesky 
    F = Hermitian(float(X'X))
    p.chol = updatechol(p.chol, F)

    # reset scratch
    p.scratchm1 = similar(X)
    p.scratchm2 = similar(p.chol.factors)

    isempty(trm.model.rr.wts) ? delbeta!(p, X, trm.model.rr.y) : delbeta!(p, X, trm.model.rr.y, trm.model.rr.wts)
    updateμ!(trm.model.rr, linpred(p, X))
    # installbeta is ommited
end 

updatechol(::CholeskyPivoted, F::AbstractMatrix{<: BlasReal}) = 
    cholesky!(F, Val(true), tol = -one(eltype(F)), check = false)

updatechol(::Cholesky, F::AbstractMatrix{<: BlasReal}) = cholesky!(F)

# Backend for GeneralizedLinearModel
function deviance(trm::TableRegressionModel{<: GeneralizedLinearModel{<: GlmResp{<: GLM.FPVector, <: GLM.UnivariateDistribution, <: Link}}}, exclude;
                    verbose::Bool = false, 
                    maxiter::Integer = 30, 
                    minstepfac::Real = 0.001,
                    atol::Real = 1e-6, 
                    rtol::Real = 1e-6, 
                    kwargs...)

    # Check arguments
    maxiter >= 1       || throw(ArgumentError("maxiter must be positive"))
    0 < minstepfac < 1 || throw(ArgumentError("minstepfac must be in (0, 1)"))

    # Extract fields and set convergence flag
    cvg, p, r = false, trm.model.pp, trm.model.rr
    lp = r.mu

    # subset X, reset beta
    X = submm(trm, exclude)
    p.beta0 = p.delbeta = repeat([0], size(X, 2))
    p.scratchbeta = similar(p.beta0)
    # cholesky 
    F = Hermitian(float(X'X))
    p.chol = updatechol(p.chol, F)

    # reset scratch
    p.scratchm1 = similar(X)
    p.scratchm2 = similar(p.chol.factors)

    # Initialize β, μ, and compute deviance
    delbeta!(p, X, GLM.wrkresp(r), r.wrkwt)
    linpred!(lp, p, X)
    updateμ!(r, lp)
    installbeta!(p)
    devold = deviance(trm.model)

    for i = 1:maxiter
        f = 1.0 # line search factor
        local dev

        # Compute the change to β, update μ and compute deviance
        try
            delbeta!(p, X, r.wrkresid, r.wrkwt)
            linpred!(lp, p, X)
            updateμ!(r, lp)
            dev = deviance(trm.model)
        catch e
            isa(e, DomainError) ? (dev = Inf) : rethrow(e)
        end

        # Line search
        ## If the deviance isn't declining then half the step size
        ## The rtol*dev term is to avoid failure when deviance
        ## is unchanged except for rouding errors.
        while dev > devold + rtol*dev
            println("ok")
            f /= 2
            f > minstepfac || error("step-halving failed at beta0 = $(p.beta0)")
            try
                updateμ!(r, linpred(p, X, f))
                dev = deviance(trm.model)
            catch e
                isa(e, DomainError) ? (dev = Inf) : rethrow(e)
            end
        end
        installbeta!(p, f)

        # Test for convergence
        verbose && println("Iteration: $i, deviance: $dev, diff.dev.:$(devold - dev)")
        if devold - dev < max(rtol*devold, atol)
            cvg = true
            break
        end
        @assert isfinite(dev)
        devold = dev
    end
    cvg || throw(ConvergenceException(maxiter))
    devold
end

# Linear prediction
function delbeta!(p::DensePredChol{<: BlasReal, <: Cholesky}, 
                    X::Matrix{<: BlasReal}, 
                    r::Vector{<: BlasReal})
    ldiv!(p.chol, mul!(p.delbeta, transpose(X), r))
    p
end
# β = (X'X)⁻¹X'y

function delbeta!(p::DensePredChol{<: BlasReal, <: CholeskyPivoted}, 
                    X::Matrix{<: BlasReal}, 
                    r::Vector{<: BlasReal})
    ch = p.chol
    beta = mul!(p.delbeta, adjoint(X), r)
    rnk = rank(ch)
    if rnk == length(beta)
        ldiv!(ch, beta)
    elseif size(ch.factors) == (0, 0) # Rank undtermined when the matrix is 0x0
        ldiv!(ch, beta)
    else
        permute!(beta, ch.piv)
        for k = (rnk + 1):length(beta)
            beta[k] = -zero(T)
        end
        LAPACK.potrs!(ch.uplo, view(ch.factors, 1:rnk, 1:rnk), view(beta, 1:rnk))
        invpermute!(beta, ch.piv)
    end
    p
end

# exterior weights
function delbeta!(p::DensePredChol{<: BlasReal, <: Cholesky}, 
                    X::Matrix{<: BlasReal}, 
                    r::Vector{<: BlasReal}, 
                    wt::Vector{<: BlasReal})
    scr = mul!(p.scratchm1, Diagonal(wt), X)
    cholesky!(Hermitian(mul!(cholfactors(p.chol), transpose(scr), X), :U))
    mul!(p.delbeta, transpose(scr), r)
    ldiv!(p.chol, p.delbeta)
    p
end

function delbeta!(p::DensePredChol{<: BlasReal, <: CholeskyPivoted}, 
                    X::Matrix{<: BlasReal}, 
                    r::Vector{<: BlasReal}, 
                    wt::Vector{<: BlasReal})
    cf = cholfactors(p.chol)
    piv = p.chol.piv
    cf .= mul!(p.scratchm2, adjoint(mul!(p.scratchm1, Diagonal(wt), X)), X)[piv, piv]
    cholesky!(Hermitian(cf, Symbol(p.chol.uplo)))
    ldiv!(p.chol, mul!(p.delbeta, transpose(p.scratchm1), r))
    p
end

# experimental
function delbeta!(p::SparsePredChol{<: BlasReal}, 
                    X::Matrix{<: BlasReal}, 
                    r::Vector{<: BlasReal}, 
                    wt::Vector{<: BlasReal})
    scr = mul!(p.scratch, Diagonal(wt), X)
    XtWX = X'*scr
    c = p.chol = cholesky(Symmetric{eltype(XtWX),typeof(XtWX)}(XtWX, 'L'))
    p.delbeta = c \ mul!(p.delbeta, adjoint(scr), r)
end

# skip updating beta0
linpred(p::LinPred, X::Matrix, f::Real = 1.0) = 
    linpred!(Vector{eltype(p.X)}(undef, size(p.X, 1)), p, X, f)

function linpred!(out, p::LinPred, X::Matrix, f::Real = 1.0)
    mul!(out, X, iszero(f) ? p.beta0 : broadcast!(muladd, p.scratchbeta, f, p.delbeta, p.beta0))
end

# Create nestedmodels
"""
    nestedmodels(trm::TableRegressionModel{<: LinearModel}; null::Bool = true, <keyword arguments>)
    nestedmodels(trm::TableRegressionModel{<: GeneralizedLinearModel}; null::Bool = true, <keyword arguments>)

    nestedmodels(::Type{LinearModel}, formula, data; null::Bool = true, <keyword arguments>)
    nestedmodels(::Type{GeneralizedLinearModel}, formula, data, distr::UnivariateDistribution, link::Link = canonicallink(d); null::Bool = true, <keyword arguments>)

Generate nested nested models `NestedModels` from a model or formula and data.

The null model will be a model with at least one factor (including intercept) if the link function does not allow factors to be 0 (factors in denominators) or the keyword argument `null` is false (default value is true).
* `InverseLink` for `Gamma`
* `InverseSquareLink` for `InverseGaussian`
* `LinearModel` fitted with `CholeskyPivoted` when `dropcollinear = true`
Otherwise, it will be an empty model.
"""
function nestedmodels(trm::M; null::Bool = true, kwargs...) where {M <: TableRegressionModel{<: LinearModel}}
    null = null && isnullable(trm.model.pp.chol) || (@warn "Empty model is not allowed due to `CholeskyPivoted`"; false)
    assign = unique(asgn(formula(trm)))
    pop!(assign)
    dropcollinear, range = null ? (false, union(0, assign)) : (true, assign)
    wts = trm.model.rr.wts
    trms = map(range) do id
        # create sub-formula, modify schema, create mf and mm
        mf, mm = subtablemodel(trm, id)
        y = response(mf)
        TableRegressionModel(fit(trm.mf.model, mm.m, y; wts, dropcollinear, kwargs...), mf, mm)
    end
    NestedModels(trms..., trm)
end

function nestedmodels(trm::M; null::Bool = true, kwargs...) where {M <: TableRegressionModel{<: GeneralizedLinearModel}}
    null = null && isnullable(trm.model) && isnullable(trm.model.pp.chol) || (@warn "Empty model is not allowed due to `CholeskyPivoted`"; false) 
    distr = trm.model.rr.d
    link = typeof(trm.model.rr).parameters[3]()
    wts = trm.model.rr.wts
    offset = trm.model.rr.offset
    assign = unique(trm.mm.assign)
    pop!(assign)
    range = null ? union(0, assign) : assign
    trms = map(range) do id
        # create sub-formula, modify schema, create mf and mm
        mf, mm = subtablemodel(trm, id)
        y = response(mf)
        TableRegressionModel(fit(trm.mf.model, mm.m, y, distr, link; wts, offset, kwargs...), mf, mm)
    end
    NestedModels(trms..., trm)
end

nestedmodels(::Type{LinearModel}, formula::FormulaTerm, data; null::Bool = true, kwargs...) = 
    nestedmodels(lm(formula, data; kwargs...); null, kwargs...)
nestedmodels(::Type{GeneralizedLinearModel}, formula::FormulaTerm, data, 
                distr::UnivariateDistribution, link::Link = canonicallink(distr); 
                null::Bool = true, kwargs...) = 
    nestedmodels(glm(formula, data, distr, link; kwargs...); null, kwargs...)

# backend for implementing nestedmodels or dropterms (in future)
function subtablemodel(trm::TableRegressionModel, id; reschema::Bool = false)
    # create sub-formula, modify schema, create mf and mm
    f = formula(trm)
    subf = subformula(f.lhs, f.rhs, id; reschema)
    if reschema
        #=
        terms = setdiff(getterms(f.rhs), getterms(subf.rhs))
        filter!(!=(Symbol("1")), terms)
        pair = collect(pairs(trm.mf.data))
        filter!(x->!in(x.first, terms), pair)
        =#
        mf = ModelFrame(subf, trm.mf.data; 
                        model = trm.mf.model, contrasts = extract_contrasts(f))
    else
        #schema = deepcopy(trm.mf.schema)
        #for term in terms
        #    pop!(schema.schema, Term(term))
        #end
        mf = ModelFrame(subf, schema, trm.mf.data, trm.mf.model)
    end
    mm = ModelMatrix(mf)
    mf, mm
end

# Null model for CholeskyPivoted is unstable
## For nestedmodels
isnullable(::CholeskyPivoted) = false
isnullable(::Cholesky) = true
isnullable(::InverseLink) = false
isnullable(::InverseSquareLink) = false
isnullable(::Link) = true

## For deviances
isnullable(::LinearModel) = true
isnullable(model::GeneralizedLinearModel) = isnullable(typeof(model.rr).parameters[3]())