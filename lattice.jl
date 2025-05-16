"""
Abstract type for a Bravais lattice. 

Should implement [`kprimtrans`](@ref), [`buildrings`](@ref) and [`hspoints`](@ref).
"""
abstract type Lattice end

"Return a list of the reciprocal lattice vectors within `n` shells in G-space, and a dictionary listing their indices."
function buildrings(lat::Lattice, n::Integer)::(Vector{Vector{Int8}},Dict{Vector{Int8},Int}) end

"Return a list of high symmetry points of the BZ in k-space."
function hspoints(lat::Lattice)::Vector{Vector{Float64}} end

"Return a matrix whose columns are the primitive G vectors."
function kprimtrans(lat::Lattice)::Matrix{Float64} end

"Convert G-space vector to k-space"
function kpointat(lat::Lattice, kG::Vector{Int8})::Vector{Float64}
    kprimtrans(lat) * kG
end

"Sample `n` points along the high symmetry line."
function samplehs(lat::Lattice, n::Integer)::Vector{Vector{Float64}}
    verts = copy(hspoints(lat))
    push!(verts, verts[1])
    diffs = diff(verts)
    dists = norm.(diffs)
    dirs = diffs ./ dists
    cumdists = [0; cumsum(dists)]

    samples = Vector{Vector{Float64}}(undef, n)

    for (i, t) in enumerate(LinRange(0, cumdists[end], n))
        if t > cumdists[2]
            dirs = dirs[2:end]
            cumdists = cumdists[2:end]
            verts = verts[2:end]
        end

        samples[i] = verts[1] + (t - cumdists[1]) * dirs[1]
    end

    samples
end

struct TriangleLattice <: Lattice
    l::Float64
    lk::Float64
    primtrans::Matrix{Float64}
    adjacents::Vector{Vector{Int8}}

    "Construct a triangular lattice with lattice constant `l`."
    function TriangleLattice(l::AbstractFloat)
        lk = 4π/√3/l
        primtrans = [[lk, 0] [lk/2, √3/2*lk]]
        adjacents = [[-1,1],[0,1],[1,0],[1,-1],[0,-1],[-1,0]]
        return new(l,lk,primtrans,adjacents)
    end
end

function buildrings(lat::TriangleLattice, n::Integer)::Tuple{Vector{Vector{Int8}},Dict{Vector{Int8},Int}}
    partial_lattice = [[j-k, k] for j in 0:n for k in 0:j-1]
    rot = [0 -1; 1 1]
    full_lattice = [[0,0]]
    keydict = Dict([0,0] => 1)
    counter = 2
    for t in 0:5
        for pt in partial_lattice
            newpt = rot^t * pt
            push!(full_lattice, newpt)
            keydict[newpt] = counter
            counter += 1
        end
    end
    (full_lattice, keydict)
end

function hspoints(lat::TriangleLattice)::Vector{Vector{Float64}}
    l = lat.lk
    [[0,0], [l/2,l/(2*sqrt(3))], [l/2, 0]]
end

function isadjacent(lat::TriangleLattice, ind1::Vector{<:Integer}, ind2::Vector{<:Integer})::Bool
    (ind2 - ind1) in lat.adjacents
end

kprimtrans(lat::TriangleLattice) = lat.primtrans