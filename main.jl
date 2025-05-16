using Plots

include("model.jl")
include("ed.jl")

### PARAMETERS

const vsl = 30.0
const v0 = -30.0
const l = 30.0

const dgate = 5.0
const epsilon = 10.0

const extent = [4 1; 1 4]
const phis = [0.0, 0.0]
const nparticles = 5
const nvals = 10


### INTERACTION POTENTIAL

const u0 = 9.05 * dgate / epsilon
function potential(q)
    if iszero(q)
        return u0
    else
        d2q2 = dgate*dgate*(q[1]*q[1] + q[2]*q[2]);
        return u0 / (1 + d2q2/(3 + d2q2/(5 + d2q2/7)))
    end
end




### MAIN

function run()
    continuum = BLGContinuumModel{Float64}(-v0)
    superlattice = TriangleLattice(l)
    slpot = Diagonal([0.3, 0.3, 1.0, 1.0] .* vsl)
    model = SuperlatticeModel(continuum, slpot, superlattice, 3)
    edlat = EDLattice(superlattice, extent, phis)
    bd = EDBandData(edlat, [model], [75])
    ed_run(bd, Matrix{Bool}(I,1,1), [nparticles], potential, nvals)
end

energies = run()
scatter(vcat([[(i,e) for e in energies[i]] for i in 1:length(energies)]...))
gui()
readline()