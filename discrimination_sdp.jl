include("jump_utils.jl") 
using MosekTools          
using LinearAlgebra
using Random
using Mosek      
using LinearAlgebra: Diagonal, kron
using JuMP
using Base.Threads: @spawn, nthreads




###############################################################################
# 1)  PPT relaxation
###############################################################################
"""
    discrimination_sdp_ppt(states, probabilities, dA, dB; verbose=false) -> (optval, M)

Solve the minimum-error state discrimination problem under the PPT relaxation.

This builds a single POVM {M_λ} on H_A ⊗ H_B with:
- completeness: Σ_λ M_λ = I_{dA·dB}
- positivity:   M_λ ⪰ 0
- PPT on A:     (T_A ⊗ I_B)(M_λ) ⪰ 0

and maximizes  Σ_λ p_λ Tr[ ρ_λ M_λ ].

Arguments
- `states::Vector{<:AbstractMatrix}`: density operators ρ_λ of size (dA*dB)×(dA*dB).
- `probabilities::Vector{Float64}`: nonnegative priors p_λ, typically summing to 1.
- `dA::Int`, `dB::Int`: local dimensions.
- `verbose::Bool=false`: if true, shows solver output.

Returns
- `optval::Float64`: optimal PPT success probability.
- `M::Vector{Matrix{ComplexF64}}`: optimal measurement elements M_λ.
"""

function discrimination_sdp_ppt(
        states::Vector{<:AbstractMatrix}, probabilities::Vector{Float64},
        dA::Int, dB::Int;  verbose::Bool=false)

    N   = length(states)
    dim = dA*dB

    model = Model(Mosek.Optimizer) 
    # variables  M_λ ∈ H⁺(dim)
    M = [ @variable(model, [1:dim, 1:dim] in HermitianPSDCone(),
                base_name = "M_$λ")
        for λ in 1:N ]

    @constraint(model, sum(M[λ] for λ in 1:N) .== I(dim))
    for λ in 1:N
        Pt = p_transpose(M[λ], [dA, dB], [1])   # partial transpose on subsystem A
        @constraint(model, Hermitian(Pt) in HermitianPSDCone())
    end
    @expression(model, obj,
        sum(probabilities[λ] * real(trace_product(states[λ], M[λ])) for λ in 1:N))
    @objective(model, Max, obj)

    if verbose
        unset_silent(model)
    else
        set_silent(model)
    end
    optimize!(model)

    return objective_value(model), [value.(M[λ]) for λ in 1:N]
end

###############################################################################
# 2)  seesaw 1R-LOCC
###############################################################################
"""
    discrimination_seesaw_1RLOCC(rho_list, p_list, dA, dB, m;
                                 seeds=30, iterations=100, tol=1e-6,
                                 verbose=false, threads=1)
        -> (best_obj, A, B, seed)

Alternating optimization (seesaw) for one-round LOCC (A → B).

Model
- Alice performs a POVM {A_a}_{a=1..m} on H_A.
- Given Alice's message a, Bob uses a POVM {B_{a,λ}}_{λ=1..N} on H_B with
  Σ_λ B_{a,λ} = I_B for each fixed a.
- Objective: maximize Σ_λ p_λ Tr[ ρ_λ Σ_a (A_a ⊗ B_{a,λ}) ].

The routine alternates:
1) optimize Bob given Alice,
2) optimize Alice given Bob,
until the objective changes by less than `tol` or `iterations` are reached.
Multiple random seeds are tried, optionally in parallel.

Arguments
- `rho_list::Vector{<:AbstractMatrix}`: states ρ_λ of size (dA*dB)×(dA*dB).
- `p_list::Vector{Float64}`: priors p_λ.
- `dA::Int`, `dB::Int`: local dimensions.
- `m::Int`: number of Alice outcomes (message alphabet size).
- `seeds`: integer number of seeds or an explicit vector of seeds.
- `iterations::Int=100`: max seesaw iterations per seed.
- `tol::Float64=1e-6`: stop if objective change is below this.
- `verbose::Bool=false`: print solver logs for the inner SDPs.
- `threads::Int=1`: maximum number of seeds optimized concurrently.

Returns
- `best_obj::Float64`: best success probability found across seeds.
- `A::Vector{Matrix{ComplexF64}}`: Alice POVM with length m, each dA×dA PSD.
- `B::Vector{Vector{Matrix{ComplexF64}}}`: for each a, a vector {B_{a,λ}} over λ
  of length N, each dB×dB PSD, and Σ_λ B_{a,λ} = I_B.
- `seed::Int`: seed that achieved `best_obj`.
"""
function discrimination_seesaw_1RLOCC(
    rho_list::Vector{<:AbstractMatrix},
    p_list::Vector{Float64},
    dA::Int, dB::Int, m::Int;
    seeds::Union{Integer,AbstractVector}=30,
    iterations::Int = 100,
    tol::Float64    = 1e-6,
    verbose::Bool   = false,                    
    threads::Int = 1)

    numλ    = length(p_list)
    seedvec = seeds isa Integer ? collect(0:seeds-1) : collect(seeds)

    C = max(1, threads)        # guard against 0 or negatives
    sem = Base.Semaphore(C)

    Results = Vector{Tuple{Float64,Any,Any,Int}}(undef, length(seedvec))

    # tiny local builder to avoid repeating verbosity lines
    build_model = function()
        model = Model(Mosek.Optimizer)
        if verbose
            unset_silent(model)
        else
            set_silent(model)
        end
        return model
    end

    nA = m #number of POVM alements of Alice is equal to the message
    IdA = Diagonal(ones(ComplexF64, dA))
    IdB = Diagonal(ones(ComplexF64, dB))

    @sync for (idx, seed) in enumerate(seedvec)
        Base.acquire(sem)
        @spawn begin
            try
                Random.seed!(seed)

                # initial POVMs
                A = random_povm(dA, nA)                     # Alice
                B = [random_povm(dB, numλ) for _ in 1:nA]   # Bob per a

                modelA = build_model()
                modelB = build_model()

                prev_obj = -Inf
                for _ in 1:iterations
                    # Bob step: optimize B with A fixed
                    JuMP.empty!(modelB)
                    Bvars = Array{Any}(undef, nA, numλ)
                    for a in 1:nA, λ in 1:numλ
                        Bvars[a, λ] = @variable(modelB, [1:dB, 1:dB] in HermitianPSDCone())
                    end
                    for a in 1:nA
                        @constraint(modelB, sum(Bvars[a, λ] for λ in 1:numλ) .== IdB)
                    end
                    @expression(modelB, objB,
                        sum(p_list[λ] * real(trace_product(rho_list[λ], kron(A[a], Bvars[a, λ])))
                            for λ in 1:numλ, a in 1:nA))
                    @objective(modelB, Max, objB)
                    optimize!(modelB)
                    B = [[value.(Bvars[a, λ]) for λ in 1:numλ] for a in 1:nA]

                    # Alice step: optimize A with B fixed
                    JuMP.empty!(modelA)
                    Avars = Vector{Any}(undef, nA)
                    for a in 1:nA
                        Avars[a] = @variable(modelA, [1:dA, 1:dA] in HermitianPSDCone())
                    end
                    @constraint(modelA, sum(Avars[a] for a in 1:nA) .== IdA)
                    @expression(modelA, objA,
                        sum(p_list[λ] * real(trace_product(rho_list[λ], kron(Avars[a], B[a][λ])))
                            for λ in 1:numλ, a in 1:nA))
                    @objective(modelA, Max, objA)
                    optimize!(modelA)
                    A = [value.(Avars[a]) for a in 1:nA]

                    # convergence check on scalar objective
                    curr_obj = sum(
                        p_list[λ] * real(tr(rho_list[λ] *
                            sum(kron(A[a], B[a][λ]) for a in 1:nA)))
                        for λ in 1:numλ
                    )
                    if abs(curr_obj - prev_obj) < tol
                        prev_obj = curr_obj
                        break
                    end
                    prev_obj = curr_obj
                end

                Results[idx] = (prev_obj, A, B, seed)
            catch
                Results[idx] = (-Inf, nothing, nothing, seed)
            finally
                Base.release(sem)
            end
        end
    end

    return reduce((x, y) -> x[1] > y[1] ? x : y, Results)
end



###############################################################################
# 3)  seesaw NA-LOCC
###############################################################################
"""
    discrimination_seesaw_NALOCC(rho_list, p_list, dA, dB, m;
                                 seeds=30, iterations=100, tol=1e-6,
                                 verbose=false, threads=1)
        -> (best_obj, A, B, seed)

Alternating optimization for nonadaptive LOCC with two parties.

Model idea
- Alice has a POVM {A_a}_{a=1..m} on H_A.
- Bob's family of POVMs is derived from a single "mother" POVM M on H_B that
  represents a fixed, nonadaptive sequence of nC=m outcomes. Marginals of M
  define consistent per-round POVMs {B_{λ,a}} used for state discrimination.
- Objective: maximize Σ_λ p_λ Tr[ ρ_λ Σ_a (A_a ⊗ B_{λ,a}) ].

The routine alternates Bob and Alice steps, similar to `discrimination_seesaw_1RLOCC`,
but Bob's variables are tied together through the mother POVM.

Arguments
- `rho_list::Vector{<:AbstractMatrix}`: states ρ_λ of size (dA*dB)×(dA*dB).
- `p_list::Vector{Float64}`: priors p_λ.
- `dA::Int`, `dB::Int`: local dimensions.
- `m::Int`: number of Alice outcomes (message alphabet size).
- `seeds`: integer number of seeds or an explicit vector of seeds.
- `iterations::Int=100`: max seesaw iterations per seed.
- `tol::Float64=1e-6`: stop if objective change is below this.
- `verbose::Bool=false`: print solver logs for the inner SDPs.
- `threads::Int=1`: maximum number of seeds optimized concurrently.

Returns
- `best_obj::Float64`: best success probability found across seeds.
- `A::Vector{Matrix{ComplexF64}}`: Alice POVM with length m.
- `B::Matrix{Matrix{ComplexF64}}`: size (N, m). Entry `B[λ, a]` is Bob's PSD
  operator used when the true state index is λ and Alice announced a.
- `seed::Int`: seed that achieved `best_obj`.
"""

function discrimination_seesaw_NALOCC(rho_list::Vector{<:AbstractMatrix},
                   p_list::Vector{Float64},
                   dA::Int, dB::Int, m::Int;
                   seeds::Union{Integer,AbstractVector}=30,
                   iterations::Int = 100,
                   tol::Float64    = 1e-6,
                   verbose::Bool   = false,
                   threads::Int = 1)

    nA = m #number of POVM alements of Alice is equal to the message
    nC   = nA                      # Bob replies with nA outcomes
    numλ = length(p_list)
    nB   = numλ
    seedvec = seeds isa Integer ? collect(0:seeds-1) : collect(seeds)

    Results = Vector{Tuple{Float64,Any,Any,Int}}(undef, length(seedvec))

    C = max(1, threads)        # guard against 0 or negatives
    sem = Base.Semaphore(C)

    IdA = Diagonal(ones(ComplexF64, dA))
    IdB = Diagonal(ones(ComplexF64, dB))

    # small local builder to set only verbosity per model
    build_model = function()
        model = Model(Mosek.Optimizer)
        if verbose
            unset_silent(model)
        else
            set_silent(model)
        end
        return model
    end

    @sync for (idx, seed) in enumerate(seedvec)
        Base.acquire(sem)
        @spawn begin
            try
                Random.seed!(seed)

                # initial POVMs
                A = random_povm(dA, nA)                         # Alice
                B = [I(dB) / nB for _ in 1:nB, _ in 1:nA]       # Bob placeholder

                prev_obj = -Inf

                # JuMP models
                modelA = build_model()
                modelB = build_model()

                # seesaw iterations
                for _ in 1:iterations
                    # Bob step
                    JuMP.empty!(modelB)

                    # 1) Mother POVM M[b₁,…,b_{nC}]
                    dims = ntuple(_ -> nB, nC)   # (nB, nB, …, nB), length nC
                    M    = Array{Any}(undef, dims...)
                    for Ic in CartesianIndices(M)
                        name = "M_" * join(Tuple(Ic), "_")
                        M[Ic] = @variable(modelB,
                                          [i=1:dB, j=1:dB] in HermitianPSDCone(),
                                          base_name = name)
                    end

                    # 2) Completeness on Bob: Σ M = I_dB
                    @constraint(modelB, sum(M[Ic] for Ic in CartesianIndices(M)) .== IdB)

                    # 3) Per round POVMs B[b,c] from marginals of M
                    @expression(modelB, B_vars[b=1:nB, c=1:nC],
                        sum(M[Ic] for Ic in CartesianIndices(M) if Ic[c] == b))

                    # 4) Bob objective with Alice A fixed
                    @expression(modelB, objB,
                        sum(p_list[λ] *
                            real(trace_product(rho_list[λ], kron(A[a], B_vars[λ, a])))
                            for λ in 1:numλ, a in 1:nA))
                    @objective(modelB, Max, objB)

                    optimize!(modelB)

                    # numeric B for Alice step
                    B = [value.(B_vars[b, a]) for b in 1:nB, a in 1:nA]

                    # Alice step
                    JuMP.empty!(modelA)
                    Avars = Vector{Any}(undef, nA)
                    for a in 1:nA
                        Avars[a] = @variable(modelA,
                                             [i=1:dA, j=1:dA] in HermitianPSDCone(),
                                             base_name = "A_$(a)")
                    end
                    @constraint(modelA, sum(Avars[a] for a in 1:nA) .== IdA)

                    @expression(modelA, objA,
                        sum(p_list[λ] *
                            real(trace_product(rho_list[λ], kron(Avars[a], B[λ, a])))
                            for λ in 1:numλ, a in 1:nA))
                    @objective(modelA, Max, objA)
                    optimize!(modelA)

                    # update Alice POVM
                    A = [value.(Avars[a]) for a in 1:nA]

                    # convergence check
                    curr_obj = sum(p_list[λ] *
                                   real(tr(rho_list[λ] * kron(A[a], B[λ, a])))
                                   for λ in 1:numλ, a in 1:nA)
                    if abs(curr_obj - prev_obj) < tol
                        prev_obj = curr_obj
                        break
                    end
                    prev_obj = curr_obj
                end

                Results[idx] = (prev_obj, A, B, seed)
            catch
                Results[idx] = (-Inf, nothing, nothing, seed)
            finally
                Base.release(sem)
            end
        end
    end

    # return best seed
    return reduce((x, y) -> x[1] > y[1] ? x : y, Results)
end


###############################################################################
# 4)  helper function symmetric extension
###############################################################################


"""
    permute_R(R, σA, dA, n, dB) -> AbstractMatrix

Permute the blocks of an operator R acting on (A₁,…,Aₙ,B) ⊗ (A₁,…,Aₙ,B) by
relabeling only the A registers on both bra and ket sides, keeping B fixed.

Arguments
- `R::AbstractMatrix`: a matrix of size (dA^n·dB)×(dA^n·dB).
- `σA::AbstractVector{<:Integer}`: permutation of 1:n that reorders A₁,…,Aₙ.
- `dA::Int`: dimension of each A subsystem.
- `n::Int`: number of A subsystems.
- `dB::Int`: dimension of B.

Returns
- The permuted matrix with B untouched and A registers permuted by `σA`.
"""
function permute_R(R::AbstractMatrix,
                   σA::AbstractVector{<:Integer},
                   dA::Int, n::Int, dB::Int)
    σ_full = vcat(σA, n + 1)                 # keep B in place
    dims   = vcat(fill(dA, n), dB)
    Rbase  = R isa Hermitian ? parent(R) : R  # JuMP Hermitian wrapper
    return permute_operator(Rbase, σ_full, dims)
end



"""
    build_canonical_R(model; dA, dB, tuple_len, cardinality, idx_sizes)
        -> DenseAxisArray

Create JuMP PSD matrix variables R[a₁,…,a_{tuple_len}, tail…] with built-in
symmetry across equal A symbols and convenient indexing for noncanonical tuples.

Purpose
- In k-symmetric-extension hierarchies you need blocks indexed by tuples of A
  outcomes of length `tuple_len` together with trailing indices for other
  registers (for example b and c). Many tuples are equivalent under permutations
  of equal symbols. This function declares a single canonical variable per class,
  adds equality constraints for symbol-swap symmetries inside a canonical tuple,
  and exposes an accessor that permutes to match any requested tuple.

Arguments
- `model::JuMP.Model`: target model.
- `dA::Int`: local dimension of A.
- `dB::Int`: local dimension of B.
- `tuple_len::Int`: length of the A-tuple, usually k+1.
- `cardinality::Int`: number of A outcomes nA.
- `idx_sizes::AbstractVector{<:Integer}`: sizes of trailing discrete indices,
  for example `[nB, nC]` or `[nB, nB, …]`.

Returns
- `R::JuMP.Containers.DenseAxisArray`: multi-axis container whose element
  `R[a₁,…,a_{tuple_len}, tail…]` is a JuMP PSD matrix variable of size
  (dA^tuple_len·dB)×(dA^tuple_len·dB). Noncanonical tuples are resolved by
  applying the correct permutation to the stored canonical variable.

"""
function build_canonical_R(model;
                           dA::Int,
                           dB::Int,
                           tuple_len::Int,
                           cardinality::Int,
                           idx_sizes::AbstractVector{<:Integer})  # e.g. [4,3,2,3]
                           
    D         = dA^tuple_len * dB
    tail_axes = map(s -> 1:s, idx_sizes)

    Rdict = Dict{Vector{Int}, JuMP.Containers.DenseAxisArray}()

    # 1. declare variables per canonical a-tuple
    for canon in canonical_tuples(cardinality, tuple_len)
        store = Array{Any}(undef, (idx_sizes...)...)           # shape = idx_sizes
        for idxs in Iterators.product(tail_axes...)            # idxs is e.g. (b,c,d,e)
            name = "R_" * join(canon, "") * "_" * join(string.(Tuple(idxs)), "_")
            store[idxs...] = @variable(model,
                                       [1:D, 1:D] in HermitianPSDCone(),
                                       base_name = name)
        end
        Rdict[canon] = JuMP.Containers.DenseAxisArray(store, tail_axes...)

        # 2. symmetry constraints among equal symbols inside canon
        blocks = Dict{Int, Vector{Int}}()
        for (pos, sym) in enumerate(canon)
            push!(get!(blocks, sym, Int[]), pos)
        end
        for pos in values(blocks)
            length(pos) <= 1 && continue
            hub = pos[1]
            for q in pos[2:end]
                σ = collect(1:tuple_len); σ[hub], σ[q] = σ[q], σ[hub]
                for idxs in Iterators.product(tail_axes...)
                    @constraint(model,
                        Rdict[canon][idxs...] .==
                        permute_R(Rdict[canon][idxs...], σ, dA, tuple_len, dB))
                end
            end
        end
    end

    # 3. getter for any a-tuple at any trailing index
    getR = let Rdict = Rdict, dA = dA, dB = dB, tuple_len = tuple_len
        function (t::NTuple{N,Int} where N, idxs::Vararg{Int})
            canon_vec = sort(collect(t))
            Rc        = Rdict[canon_vec][idxs...]
            if t == Tuple(canon_vec)
                return Rc
            end
            σ, used = similar(canon_vec), falses(tuple_len)
            for i in 1:tuple_len
                j = findfirst(j -> !used[j] && canon_vec[j] == t[i], 1:tuple_len)
                σ[i] = j; used[j] = true
            end
            return permute_R(Rc, σ, dA, tuple_len, dB)
        end
    end

    # 4. materialize full DenseAxisArray over (a..., trailing indices...)
    a_axes = ntuple(_ -> 1:cardinality, tuple_len)
    data   = Array{Any}(undef, (fill(cardinality, tuple_len)..., idx_sizes...)...)
    for a in Iterators.product(a_axes...)
        for idxs in Iterators.product(tail_axes...)
            data[a..., idxs...] = getR(Tuple(a), idxs...)
        end
    end

    return JuMP.Containers.DenseAxisArray(data, (a_axes..., tail_axes...)...)
end


###############################################################################
# 5)  1R-LOCC hierarchy
###############################################################################
"""
    discrimination_hierarchy_1RLOCC(states, probabilities, dA, dB, m, k; verbose=false)
        -> (optval, R)

k-symmetric-extension SDP upper bound for one-round LOCC (A → B).


Arguments
- `states::Vector{<:AbstractMatrix}`, `probabilities::Vector{Float64}`: ρ_λ and p_λ.
- `dA::Int`, `dB::Int`: local dimensions.
- `m::Int`: number of Alice outcomes (message alphabet size).
- `k::Int`: level of the hierarchy.
- `verbose::Bool=false`: show solver logs.

Returns
- `optval::Float64`: upper bound on the best 1R-LOCC success probability.
- `R::DenseAxisArray`: JuMP container with the block variables.
"""



function discrimination_hierarchy_1RLOCC(states::Vector{<:AbstractMatrix},
    probabilities::Vector{Float64},
        dA::Int, dB::Int, m::Int, k::Int; verbose::Bool= false)
    # Number of states and Bob messages
    nB = length(states)

    nA = m #number of POVM alements of Alice is equal to the message
    nC = nA  # Bob replies index match

    # Identity operators
    I_A = Diagonal(ones(dA))
    I_B = Diagonal(ones(dB))

    # initiate the model
    model = Model(Mosek.Optimizer) 

    # -- Decision variables --
    # -------------------------------------------------
    #  R[ a₁ , … , a_{k} ,  b  ,  c ]       ∈ ℂ^{Rdim × Rdim}
    #  each block is Hermitian and PSD
    # -------------------------------------------------


    # build variables + symmetry ---------------------------------------------
    R = build_canonical_R(model;
            dA          = dA,
            dB          = dB,
            tuple_len   = k,
            cardinality = nA,
            idx_sizes   = [nB,nC])

 
    # --- tuple of ranges  (1:nA, 1:nA, …, 1:nA)  with length k+1 ---------------
    Ak1_ranges = ntuple(_ -> 1:nA, k)
    Ak_ranges = ntuple(_ -> 1:nA, k-1)
    # --- constraints  
    
    for a_tuple in canonical_tuples(nA, k)
        Ra1 = @expression(model, sum(R[a_tuple..., b, 1] for b in 1:nB))
        @constraint(model,Ra1 ==  kron(p_trace(Ra1, [dA^(k), dB], [2]), I_B/dB))
        for c in 2:nC
            @constraint(model,Ra1  == sum(R[a_tuple..., b, c] for b in 1:nB))
        end
    end


    for a_tuple in canonical_tuples(nA, k-1), b in 1:nB, c in 1:nC
        Rb = @expression(model, sum(R[a, a_tuple..., b, c] for a in 1:nA))
        @constraint(model, Rb == kron(I_A/dA, p_trace(Rb, [dA, dB * dA^(k-1)], [1])))
    end
    

    @constraint(model, tr(sum(R[a_tuple..., b, 1] for a_tuple in Iterators.product(Ak1_ranges...), b in 1:nB)) == dA^(k)*dB)
   
    
    # -- PPT constraints--
    # Symmetry allows me to just PPT test N A's along with B
    for a_tuple in canonical_tuples(nA, k), b in 1:nB, c in 1:nC, T_dim in 0:k-1
        Pt = p_transpose(R[a_tuple...,b,c], [dA^(k- T_dim ),(dA^(T_dim)) * dB], [2])
        @constraint(model, Hermitian(Pt)  in HermitianPSDCone())
    end



    # -- Objective: maximize success probability --
    @expression(model, Mλ[b = 1:nB],# Here we are defining Mλ = (1/|Ã|)Tr_Ã Σ_{ãa}R[ãaλa]
        (1/dA^(k-1))*p_trace(sum(R[a_tuple...,a,b,a] for a_tuple in Iterators.product(Ak_ranges...), a in 1:nA),
        [dA^(k-1), dA * dB], [1]))
    @objective(model, Max, real(sum(probabilities[b] * trace_product(states[b], Mλ[b]) for b in 1:nB)))
    
    if verbose
        unset_silent(model)
    else
        set_silent(model)
    end

    optimize!(model)
    return objective_value(model), R
end

###############################################################################
# 6)  NA-LOCC hierarchy
###############################################################################
"""
    discrimination_hierarchy_NALOCC(states, probabilities, dA, dB, m, k; verbose=false)
        -> (optval, R)

k-symmetric-extension SDP upper bound for nonadaptive LOCC.

Arguments
- `states::Vector{<:AbstractMatrix}`, `probabilities::Vector{Float64}`: ρ_λ and p_λ.
- `dA::Int`, `dB::Int`: local dimensions.
- `m::Int`: number of Alice outcomes (message alphabet size).
- `k::Int`: hierarchy level.
- `verbose::Bool=false`: show solver logs.

Returns
- `optval::Float64`: upper bound on the best NA-LOCC success probability.
- `R::DenseAxisArray`: JuMP container with the block variables.
"""
function discrimination_hierarchy_NALOCC(states::Vector{<:AbstractMatrix},
    probabilities::Vector{Float64},
        dA::Int, dB::Int, m::Int, k::Int;verbose::Bool=false)
    # Number of states and Bob messages

    nA = m #number of POVM alements of Alice is equal to the message
    nB = length(states)
    nC = nA  # Bob replies index match

    # Identity operators
    I_A = Diagonal(ones(dA))
    I_B = Diagonal(ones(dB))
    # initiate the model
    model = Model(Mosek.Optimizer) 

    # -- Decision variables --
    # -------------------------------------------------
    #  R[ a₁ , … , a_{k} ,  b  ,  c ]       ∈ ℂ^{Rdim × Rdim}
    #  each block is Hermitian and PSD
    # -------------------------------------------------


    # build variables + symmetry ---------------------------------------------
    R = build_canonical_R(model;
            dA          = dA,
            dB          = dB,
            tuple_len   = k,
            cardinality = nA,
            idx_sizes   = fill(nB, nC))


    # --- tuple of ranges  (1:nA, 1:nA, …, 1:nA)  with length k ---------------
    Ak1_ranges = ntuple(_ -> 1:nA, k)
    Ak_ranges = ntuple(_ -> 1:nA, k-1)
    B_ranges = ntuple(_ -> 1:nB, nC)

    
    for a_tuple in canonical_tuples(nA, k)
        Ra1 = @expression(model, sum(R[a_tuple..., b_tuple...] for b_tuple in Iterators.product(B_ranges...)))
        @constraint(model,Ra1 .==  kron(p_trace(Ra1, [dA^(k), dB], [2]), I_B./dB))
    end

    for  a_tuple in canonical_tuples(nA, k-1), b_tuple in Iterators.product(B_ranges...)
        Rb = @expression(model, sum(R[a, a_tuple..., b_tuple...] for a in 1:nA))
        @constraint(model, Rb .== kron(I_A./dA, p_trace(Rb, [dA, dB * dA^(k-1)], [1])))
    end
    

    @constraint(model, tr(sum(R[a_tuple..., b_tuple...] for a_tuple in Iterators.product(Ak1_ranges...), b_tuple in Iterators.product(B_ranges...))) == dA^(k)*dB)
   
    
    # -- PPT constraints--
    # Symmetry allows me to just PPT test N A's along with B
    for a_tuple in canonical_tuples(nA, k), b_tuple in Iterators.product(B_ranges...), T_dim in 0:k-1
        Pt = p_transpose(R[a_tuple...,b_tuple...], [dA^(k- T_dim ),(dA^(T_dim)) * dB], [2])
        @constraint(model, Hermitian(Pt)  in HermitianPSDCone())
    end

    B_other_axes = ntuple(_ -> 1:nB, max(nC - 1, 0))

    @expression(model, Mλ[b = 1:nB],
        (1/dA^(k-1)) * p_trace(
            sum( begin
                    # for each (ã, a) accumulate over the other nC-1 trailing axes
                    sum(parent(R[a_tuple..., a,
                        ntuple(jj -> jj == a ? b : bb[jj - (jj > a)], nC)...])
                        for bb in Iterators.product(B_other_axes...))
                end
                for a_tuple in Iterators.product(Ak_ranges...), a in 1:nA
            ),
            [dA^(k-1), dA * dB], [1]
        )
    )

    @objective(model, Max, real(sum(probabilities[b] * trace_product(states[b], Mλ[b]) for b in 1:nB)))
    if verbose
        unset_silent(model)
    else
        set_silent(model)
    end
    optimize!(model)
    return objective_value(model), R
end
