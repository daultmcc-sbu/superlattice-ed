include("model.jl")

using FLoops
using MicroCollections
using BangBang
using Combinatorics
using SparseArrays
using KrylovKit

struct EDLattice
    kpoints::Vector{Vector{Int8}}
    kindices::Dict{Vector{Int8}, Int}
    gprimtrans::Matrix{Float64}
    extent::Matrix{Int8}
    adjextent::Matrix{Int8}
    phi::Vector{Float64}

    """
    Construct a lattice of points in the BZ for a finite real-space lattice on a torus.

    `extent` is a matrix whose rows are the edges of the torus parallelogram, in units of the
    r-space primitive lattice vectors (a-space). Should be positive determinant. `phi` is a 2-element 
    vector of the inserted fluxes, 0 to 1.
    """
    function EDLattice(
        lat::Lattice, 
        extent::Matrix{<:Integer}, 
        phi::Vector{<:Real}
    )
        a, c, b, d = extent
        adjextent = [d -b; -c a]
        nsites = det(extent)
        gprimtrans = kprimtrans(lat) * adjextent ./ nsites
        kpoints = []
        kindices = Dict{Vector{Int8},Int}()
        for candidate in collect.(Iterators.product(-nsites:nsites, -nsites:nsites))
            if all(0 .<= adjextent * candidate .< nsites)
                push!(kpoints, candidate)
                kindices[candidate] = length(kpoints)
            end
        end
        new(kpoints, kindices, gprimtrans, extent, adjextent, phi)
    end
end

"Convert k point in momentum lattice basis to physical basis."
kpointat(lat::EDLattice, k::Vector{Int8})::Vector{Float64} = lat.gprimtrans * (k + lat.phi)

"""
Pull a vector back into the primitive unit cell, also returning the displacement
in reciprocal lattice basis.
"""
function pullback(lat::EDLattice, k::Vector{<:Integer})::Tuple{Vector{Int8}, Vector{Int8}}
    primoffset::Vector{Int8} = fld.((lat.adjextent * k), length(lat.kpoints))
    return (k - lat.extent * primoffset, primoffset)
end

"""
Represents the single particle data (energies and states) of a number of bands (flavors)
diagonalized on the points of an `EDLattice`.
"""
struct EDBandData{F<:AbstractFloat}
    lat::EDLattice
    energies::Array{F,2}
    states::Array{Complex{F},4} # indexing order: layer+sublattice, g, band+spin+valley, k
    # scattering::Matrix{Bool}
    gpoints::Vector{Vector{Int8}}
    gindices::Dict{Vector{Int8},Int}
end

"""
Solve single-particle bands at points of `lat`.

Each model in `models` corresponds to a flavor index (band, spin, valley). `band[i]` is the index
of the band of `models[i]` to be used.
"""
function EDBandData(
    lat::EDLattice, 
    models::Vector{<:SuperlatticeModel{F}}, 
    bands::Vector{<:Integer}, 
    # scattering::Matrix{Bool}
) where {F<:AbstractFloat}
    m1 = models[1] # reference model to extract info from
    energies = Array{F,2}(undef, length(models), length(lat.kpoints))
    states = Array{Complex{F},4}(undef, 
        m1.cbands, 
        length(m1.gpoints), 
        length(models), 
        length(lat.kpoints))
    for j in eachindex(lat.kpoints)
        for i in eachindex(models)
            res = eigen(hamiltonian(models[i], kpointat(lat, lat.kpoints[j])))
            energies[i,j] = res.values[bands[i]]
            states[:,:,i,j] = res.vectors[:,bands[i]]
        end
    end
    EDBandData{F}(lat, energies, states, m1.gpoints, m1.gindices)
end

"Calculate the total momentum (in momentum lattice basis, in the primitive unit cell) of a configuration."
function totalk(lat::EDLattice, config::BitMatrix)::Vector{Int8}
    k = [0,0]
    for (kind, ns) in enumerate(eachcol(config))
        k += lat.kpoints[kind] * sum(ns)
    end
    pullback(lat, k)[1]
end

"""
Generate a list of all configurations (Fock space basis states) with a given
total momentum and number of particles per sector.
"""
function generate_configs(
    lat::EDLattice, 
    sectors::Vector{Int}, 
    nparticles::Vector{<:Integer},
    totk::Vector{<:Integer}
)::Tuple{Vector{BitMatrix},Dict{BitMatrix,Int}}
    sector_string_vecs = map(eachindex(nparticles)) do i
        len = (sectors[i+1] - sectors[i]) * length(lat.kpoints)
        BitVector.(multiset_permutations(1:len .<= nparticles[i], len))
    end

    @floop for strings in Iterators.product(sector_string_vecs...)
        config = BitArray(undef, sectors[end], length(lat.kpoints))
        for (i, string) in enumerate(strings)
            config[(sectors[i]+1):sectors[i+1],:] = string
        end
        @reduce( configs = append!!(EmptyVector(), 
            totalk(lat,config) == totk 
                ? SingletonVector((config,)) 
                : EmptyVector()) 
        )
    end

    config_indices = Dict((config => index) for (index, config) in pairs(configs))

    (configs, config_indices)
end

"""
Stores all the data needed for an ED run in a single momentum and particle number sector,
namely the list of all allowed configurations and the potential function.
"""
struct EDContext{F <: AbstractFloat, V <: Function}
    bd::EDBandData{F}
    scattering::Matrix{Bool}
    configs::Vector{BitMatrix}
    config_indices::Dict{BitMatrix,Int}
    potential::V
end

"""
Returns a vector `ind` so that `mat[ind[i]+1:ind[i+1],ind[i]+1:ind[i+1]` are the smallest blocks
on the diagonal of `mat`.
"""
function block_indices(mat::Matrix{Bool})::Vector{Int}
    indices::Vector{Int} = [0]
    while !isempty(mat)
        i = 1
        while any(mat[i+1:end,1:i]) || any(mat[1:i,i+1:end])
            i += 1
        end
        push!(indices, indices[end] + i)
        mat = mat[i+1:end,i+1:end]
    end
    indices
end

"""
`scattering[i,j]` is `True` if scattering is allowed between flavors `i` and `j`, and should be 
symmetric, block diagonal, and have 1s down the diagonal. `nparticles[i]` is the number of particles
in the `i`th conserved flavor sector, corresponding to the `i`th block of `scattering`. `totk` is the
total momentum, and `potential` is a function of momentum difference `q` (in the PHYSICAL basis)
defining the interaction potential.
"""
function EDContext(
    bd::EDBandData{F},
    scattering::Matrix{Bool},
    nparticles::Vector{Int}, 
    totk::Vector{Int8}, 
    potential::V
)::EDContext{F,V} where {F,V}
    sectors = block_indices(scattering)
    configs, config_indices = generate_configs(bd.lat, sectors, nparticles, totk)
    EDContext(bd, scattering, configs, config_indices, potential)
end

function fermisgn(ket::BitMatrix, inds::Vector{Tuple{Int,Int}})::Int
    ket_ = ket[:]
    li = LinearIndices(ket)
    sgn = 1
    for (aind, kind) in inds
        ind = li[aind,kind]
        if isodd(sum(@view ket_[1:ind]))
            sgn = -sgn
        end
        ket_[kind] = false
    end
    sgn
end

function formfactor(
    bd::EDBandData{F}, 
    k1i::Int, alpha1::Int,
    k2i::Int, alpha2::Int,
    g::Vector{Int8}
)::Complex{F} where {F <: AbstractFloat}
    ff::Complex{F} = 0
    for bandi in size(bd.states)[3]
        for gi in eachindex(bd.gpoints)
            g_ = bd.gpoints[gi]
            g1_ = g_ + g
            if haskey(bd.gindices, g1_)
                g1i = bd.gindices[g1_]
                ff += dot(
                    bd.states[:, g1i, alpha1, k1i], 
                    bd.states[:, gi,  alpha2, k2i]
                )
            end
        end
    end

    ff
end

"""
Construct a single column of the many body Hamiltonian. Can be thought of as the
result of applying the Hamiltonian to the Fock space basis vector `config`.
"""
function mbhamiltonian_column(
    ctx::EDContext{F}, configind::Int
)::SparseVector{Complex{F},Int} where {F}
    config = ctx.configs[configind]
    nflavors = size(ctx.scattering)[1]

    out = spzeros(Complex{F}, length(ctx.configs))
    occupied = findall(config)

    # kinetic energy
    out[configind] = sum(@view ctx.bd.energies[occupied])

    # interaction
    # bra = BitMatrix(undef, size(config)) # allocate ahead of time
    for (i, state1) in pairs(occupied), state2 in @view occupied[i+1:end]
        alpha1 = state1[1]
        k1i = state1[2]
        k1 = ctx.bd.lat.kpoints[k1i]

        alpha2 = state2[1]
        k2i = state2[2]
        k2 = ctx.bd.lat.kpoints[k2i]

        # loop over flavors to scatter into
        for beta1 in 1:nflavors
            # skip if trying to scatter out of flavor sector
            if !ctx.scattering[beta1,alpha1] continue end
            for beta2 in 1:nflavors
                if !ctx.scattering[beta2,alpha2] continue end

                # loop over momentum transfer
                for q in ctx.bd.lat.kpoints
                    # calculate new momenta
                    k1new, g1 = pullback(ctx.bd.lat, k1 - q)
                    k2new, g2 = pullback(ctx.bd.lat, k2 + q)

                    # skip if trying to scatter both particles into same state
                    if k1new == k2new && beta1 == beta2
                        continue
                    end

                    k1newi = ctx.bd.lat.kindices[k1new]
                    k2newi = ctx.bd.lat.kindices[k2new]

                    # skip if the new states are already occupied
                    if config[beta1, k1newi] || config[beta2, k2newi]
                        continue
                    end

                    # build new configuration
                    # bra .= config
                    bra = copy(config)
                    bra[alpha1, k1i] = false
                    bra[alpha2, k2i] = false
                    bra[beta1, k1newi] = true
                    bra[beta2, k2newi] = true

                    # @show (config, totalk(ctx.bd.lat, config))
                    # @show (bra, totalk(ctx.bd.lat, bra))

                    # calculate interaction matrix element
                    matel = zero(Complex{F})
                    for g in ctx.bd.gpoints
                        ff1 = @inline formfactor(ctx.bd, k1newi, beta1, k1i, alpha1, g + g1)
                        ff2 = @inline formfactor(ctx.bd, k2newi, beta2, k2i, alpha2, g + g2)
                        qphys = ctx.bd.lat.gprimtrans * (q + ctx.bd.lat.extent * g)
                        matel += ctx.potential(qphys) * ff1 * ff2
                    end
                    matel *= @inline fermisgn(config, [(alpha1, k1i), (alpha2, k2i)])
                    matel *= @inline fermisgn(bra, [(beta1, k1newi), (beta2, k2newi)])

                    out[ctx.config_indices[bra]] = matel
                end
            end
        end
    end

    out
end

"""
Computes the product of the many body Hamiltonian with an arbitrary Fock space
vector, for use in on-the-fly diagonalization.
"""
function mbhamiltonian_product(
    ctx::EDContext{F}, vec::Vector{Complex{F}}
)::Vector{Complex{F}} where {F}
    out = zeros(Complex{F}, length(vec))
    for i in eachindex(vec)
        out .+= vec[i] .* mbhamiltonian_column(ctx, i)
    end
    out
end

"Constructs the full many body Hamiltonian."
function construct_mbhamiltonian(
    ctx::EDContext{F}
)::SparseMatrixCSC{Complex{F},Int} where {F}
    nconfigs = length(ctx.configs)

    cols = Vector{SparseVector{Complex{F},Int}}(undef, nconfigs)
    @Threads.threads for i in 1:nconfigs
        cols[i] = mbhamiltonian_column(ctx, i)
    end

    colptr::Vector{Int} = [1]
    rowval::Vector{Int} = []
    nzval::Vector{Complex{F}} = []

    # for i in 1:nconfigs
    #     col = mbhamiltonian_column(ctx, i)
    for col in cols
        append!(rowval, col.nzind)
        append!(nzval, col.nzval)
        push!(colptr, colptr[end] + length(col.nzind))
    end

    SparseMatrixCSC(nconfigs, nconfigs, colptr, rowval, nzval)
end

function diagonalize(ctx::EDContext{F}, nvals::Integer; onthefly::Bool=false) where {F}
    if onthefly
        x0 = rand(Complex{F}, length(ctx.configs))
        energies, vecs, = eigsolve(
            v -> mbhamiltonian_product(ctx, v), x0,
            nvals, :SR, ishermitian=true, verbosity=0, tol=1e-8)
    else
        ham = construct_mbhamiltonian(ctx)
        energies, vecs, = eigsolve(ham, nvals, :SR, ishermitian=true, verbosity=0)
    end

    (energies, vecs)
end

"Diagonalize in each momentum sector."
function ed_run(bd::EDBandData{F}, scattering::Matrix{Bool}, nparticles::Vector{Int}, potential::V, nvals::Integer; onthefly::Bool=false) where {F,V}
    energies = Vector(undef, length(bd.lat.kpoints))
    for (i, totk) in pairs(bd.lat.kpoints)
        ctx = EDContext(bd, scattering, nparticles, totk, potential)
        e, = diagonalize(ctx, nvals; onthefly)
        energies[i] = e
    end
    energies
end