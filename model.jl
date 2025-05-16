using LinearAlgebra

include("lattice.jl")

abstract type Model{F<:AbstractFloat} end

function hamiltonian end
(hamiltonian(model::Model{F}, x::AbstractFloat, y::AbstractFloat)::Matrix{Complex{F}}) where {F<:AbstractFloat} = hamiltonian(model, [x,y])
(hamiltonian(model::Model{F}, pt::AbstractVector{<:AbstractFloat})::Matrix{Complex{F}}) where {F<:AbstractFloat} = hamiltonian(model, pt[1], pt[2])
# hamiltonian(model::Model, xs::AbstractVector{<:AbstractFloat}, ys::AbstractVector{<:AbstractFloat}) = (hamiltonian(model, x, y))

function bands end

# function spectrum(model::Model, x::AbstractFloat, y::AbstractFloat)
#     eigvals(hamiltonian(model, x, y))
# end

abstract type LatticeModel{F} <: Model{F} end

function lattice end

struct SuperlatticeModel{F<:AbstractFloat, L<:Lattice,  M<:Model{F}} <: LatticeModel{F}
    continuum::M
    superlattice::L
    gpoints::Vector{Vector{Int8}}
    gindices::Dict{Vector{Int8},Int}
    kpoints::Vector{Vector{Float64}}
    cbands::Int
    totbands::Int
    slham::Matrix{Complex{F}}
    function SuperlatticeModel(continuum::M, slpot::AbstractMatrix, superlattice::L, rings::Integer) where {F<:AbstractFloat, L<:Lattice, M<:Model{F}}
        cbands = bands(continuum)
        # if eltype(slpot) != dtype(continuum) error("slpot has incorrect eltype") end
        if size(slpot) != (cbands, cbands) error("slpot has incorrect size") end
        gpoints, gindices = buildrings(superlattice, rings)
        kpoints = kpointat.(Ref(superlattice), gpoints)
        totbands = length(gpoints) * cbands
        slham = zeros(F, (totbands, totbands))
        for (i, g1) in enumerate(gpoints)
            for (j, g2) in enumerate(gpoints)
                if isadjacent(superlattice, g1, g2)
                    slham[cbands*(i-1)+1:cbands*i,cbands*(j-1)+1:cbands*j] = slpot
                end
            end
        end
        new{F,L,M}(continuum, superlattice, gpoints, gindices, kpoints, cbands, totbands, slham)
    end
end

bands(model::SuperlatticeModel) = model.totbands
lattice(model::SuperlatticeModel) = model.superlattice

function hamiltonian(model::SuperlatticeModel, x::Float64, y::Float64)
    ham = copy(model.slham)
    for (i, pt) in enumerate(model.kpoints)
        rn = model.cbands*(i-1)+1:model.cbands*i
        ham[rn,rn] = hamiltonian(model.continuum, x+pt[1], y+pt[2])
    end
    ham
end

struct BLGContinuumModel{T<:AbstractFloat} <: Model{T}
    u::T
    t::T
    v::T
    v3::T
    v4::T
    function BLGContinuumModel{T}(u::T; t::T=380.0, v::T=673.0, v3::T=81.0, v4::T=30.0) where {T<:AbstractFloat}
        new{T}(u, t, v, v3, v4)
    end
end

bands(model::BLGContinuumModel) = 4

function hamiltonian(model::BLGContinuumModel, x::Float64, y::Float64)
    z = complex(x, y)
    zb = complex(x, -y)
    u = model.u
    t = model.t
    v = model.v
    v3 = model.v3
    v4 = model.v4
    [-u      v*zb  -v4*zb   v3*z
      v*z   -u      t      -v4*zb
     -v4*z   t      u       v*zb
      v3*zb -v4*z   v*z     u    ]
end