using TensorOperations, JuMP, ComplexOptInterface, Mosek, MosekTools, LinearAlgebra, Combinatorics
"""
Enforce PSD constraint on the input matrix.
Uses dummy vars. to enforce PPT (not possible directly in ComplexOptInterface?).
"""
function ispsd(problem::Model, A::AbstractMatrix)
    @assert size(A,1) == size(A,2) "Matrix must be square."
    @assert length(size(A)) == 2 "Matrix must be bidimensional."

    dim = size(A, 1)
    PSD = @variable(problem, [1:dim, 1:dim] in ComplexOptInterface.HermitianPSDCone())
    @constraint(problem, A .== PSD)
end

"""
Partial trace and partial transpose operations were
taken from https://github.com/iitis/QuantumInformation.jl
"""

"""
- `ρ`: quantum state.
- `idims`: dimensions of subsystems.
- `isystems`: traced subsystems.
Return [partial trace](https://en.wikipedia.org/wiki/Partial_trace) of matrix `ρ` over the subsystems determined by `isystems`.
"""
function p_trace(ρ::AbstractMatrix, idims::Vector{<:Integer}, isystems::Vector{<:Integer})
    N = length(idims)
    size(ρ, 1) == size(ρ, 2)              ||
        throw(ArgumentError("ρ must be square."))
    prod(idims) == size(ρ, 1)             ||
        throw(ArgumentError("Product(idims) ≠ size(ρ)."))
    all(1 .≤ isystems .≤ N)               ||
        throw(ArgumentError("`isystems` out of range 1:$N"))

    # same two lines as the original helper
    dims    = reverse(idims)                         # memory-layout order
    systems = N .- isystems .+ 1                     # count from the right

    keep        = setdiff(1:N, systems)
    kept_dim    = prod(dims[keep])
    traced_dim  = prod(dims[systems])

    # reshape  →  ket_N … ket₁  bra_N … bra₁
    T = reshape(ρ, vcat(dims, dims)...)

    # move traced subsystems to the end of the ket axes
    perm_ket = vcat(keep, systems)
    perm     = vcat(perm_ket, perm_ket .+ N)
    T = permutedims(T, perm)

    # reshape into (kept, traced, kept, traced)
    T = reshape(T, kept_dim, traced_dim, kept_dim, traced_dim)

    # trace over the two `traced` axes
    result = fill(zero(eltype(ρ)), kept_dim, kept_dim)
    for t in 1:traced_dim
        result .+= T[:, t, :, t]
    end
    return result
end

"""
- `ρ`: quantum state.
- `idims`: dimensins of subsystems.
- `isystems`: transposed subsystems.
"""
function p_transpose(ρ::AbstractMatrix, idims::Vector{Int}, isystems::Vector{Int})
    dims = reverse(idims)
    systems = length(idims) .- isystems .+ 1

    if size(ρ,1)!=size(ρ,2)
        throw(ArgumentError("Non square matrix passed to ptrace"))
    end
    if prod(dims)!=size(ρ,1)
        throw(ArgumentError("Product of dimensions do not match shape of matrix."))
    end
    if maximum(systems) > length(dims) ||  minimum(systems) < 1
        throw(ArgumentError("System index out of range"))
    end

    offset = length(dims)
    tensor = reshape(ρ, [dims; dims]...)
    perm = collect(1:(2*offset))
    for s in systems
        idx1 = findfirst(x->x==s, perm)
        idx2 = findfirst(x->x==(s + offset), perm)
        perm[idx1], perm[idx2] = perm[idx2], perm[idx1]
    end
    tensor = permutedims(tensor, invperm(perm))
    reshape(tensor, size(ρ))
end

"""
Swaps system k and k+1 of X written with the tensor product structure dictated by dims
"""
function swap_neighbors(X, k, dims)
    # Swaps systems k and k+1 of rho
    n = length(dims)
    @assert 1 ≤ k < n "k must label a neighbour pair within 1…n-1"

    # map “physical” subsystem index k → axis index in the reversed layout
    r1 = n - k + 1          # axis for subsystem k
    r2 = r1 - 1             # axis for subsystem k+1

    # reshape into ket axes ⊗ bra axes, both in reverse order
    R = reshape(X, (reverse(dims)..., reverse(dims)...))

    perm_ket = collect(1:n)
    perm_ket[r1], perm_ket[r2] = perm_ket[r2], perm_ket[r1]   # swap
    order = vcat(perm_ket, perm_ket .+ n)     # do the same for bra axes

    reshape(permutedims(R, order), size(X))
end

"""
    trace_product(A, B)
Return Tr(AB).
"""

function trace_product(A::AbstractMatrix{<:Number}, B::AbstractMatrix{<:GenericAffExpr})
    # ─── sanity checks ────────────────────────────────────────────────────
    size(A, 1) == size(A, 2) ||
        throw(ArgumentError("A must be square (got $(size(A)))."))
    size(B) == size(A) ||
        throw(ArgumentError("B must have the same shape as A "
                            * "(A: $(size(A)), B: $(size(B)))."))

    n = size(A, 1)
    return sum(A[i, j] * B[j, i] for i = 1:n, j = 1:n)
end
function trace(A::AbstractMatrix{<:GenericAffExpr})
    n = size(A,1)
    @assert size(A,2) == n "must be square"
    return sum(A[i,i] for i in 1:n)
end

function random_povm(dim::Int, k::Int)
    G   = [randn(ComplexF64, dim, dim) for _ in 1:k]
    Fs  = [Gi' * Gi for Gi in G]               # PSD
    S   = sum(Fs)                              # positive-definite
    # build S^{-1/2}  (any PSD-preserving factorisation is fine)
    evals, vecs = eigen(Hermitian(S))
    Sinvhalf    = vecs * Diagonal(1 ./ sqrt.(evals)) * vecs'
    return [Sinvhalf * F * Sinvhalf for F in Fs]   # each element PSD
end

"""
    canonical_reps(d, k)

Return all the weakly-increasing k-tuples (the orbit-representatives)
drawn from the alphabet 1:d.  There are binomial(d+k-1, k) of them.
"""
function canonical_tuples(d::Int, k::Int)
    return collect(with_replacement_combinations(1:d, k))
end

"""
    all_det_assignments(num_λ::Int, nA::Int, nB::Int) -> Vector{Array{Int,3}}

Generate all deterministic post-processing maps D[λ,a,b] ∈ {0,1}, 
where for each (a,b) exactly one λ has D[λ,a,b]==1.
Returns a vector of 3D arrays of size (num_λ, nA, nB).
"""
function all_det_assignments(num_λ::Int, nA::Int, nB::Int)
    dets = Vector{Array{Int,3}}()  # container for all D arrays
    # Flatten (a,b) pairs into slots 1:(nA*nB)
    slots = ntuple(_->1:num_λ, nA*nB)
    # Iterate over every assignment of λ to each flattened slot
    for assignment in Iterators.product(slots...)
        D = zeros(Int, num_λ, nA, nB)  # start with all zeros
        # Fill exactly one λ=1 for each (a,b)
        for (i, λ) in enumerate(assignment)
            # Recover the original (a,b) from the flattened index i
            a = fld(i-1, nB) + 1
            b = (i-1) % nB + 1
            D[λ, a, b] = 1
        end
        push!(dets, D)
    end
    return dets
end



"""
    permute_operator(R, σ, dims) → Matrix

`R` is a square matrix that acts on the Hilbert space  
ℋ = ⊗_{j=1}^{m} 𝐂^{dims[j]}  (ket space)  
and we interpret it as a density operator on ℋ, i.e. it has
`2m` tensor axes (ket ⊗ bra).

`σ` is a permutation of `1:m`.  
The function applies that same permutation **simultaneously** to the ket
axes and to the bra axes.

`dims` is the *physical* list of subsystem dimensions
**(most-significant first)**.
"""
function permute_operator_old(R::AbstractMatrix,
                          σ::AbstractVector{<:Integer},
                          dims::AbstractVector{<:Integer})
    m = length(dims)
    @assert sort(σ) == collect(1:m) "σ must be a permutation of 1:m"

    # reshape to ket ⊗ bra, both in REVERSED (memory) order
    dims_mem = reverse(dims)
    T = reshape(R, (dims_mem..., dims_mem...))

    # Build new→old mapping for permutedims on the ket block.
    # "old j → new σ[j]" in physical space becomes (in memory space):
    #   old axis = to_mem(j), new axis = to_mem(σ[j])
    to_mem(i) = m - i + 1
    order_ket = Vector{Int}(undef, m)
    for j in 1:m
        order_ket[to_mem(σ[j])] = to_mem(j)
    end
    # apply same permutation to bra axes
    order = vcat(order_ket, order_ket .+ m)

    return reshape(permutedims(T, order), size(R))
end

function permute_operator(R::AbstractMatrix,
    σ::AbstractVector{<:Integer},
    dims::AbstractVector{<:Integer})
    m = length(dims)
    @assert sort(σ) == collect(1:m)

    dims_mem = reverse(dims)
    T = reshape(R, (dims_mem..., dims_mem...))

    to_mem(i) = m - i + 1
    order_ket = Vector{Int}(undef, m)
    # σ[i] = which OLD physical axis becomes NEW physical axis i
    for i in 1:m
        order_ket[to_mem(i)] = to_mem(σ[i])  # build new→old in memory indices
    end
    order = vcat(order_ket, order_ket .+ m)

    return reshape(permutedims(T, order), size(R))
end