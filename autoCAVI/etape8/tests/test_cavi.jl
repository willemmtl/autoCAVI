using Test

include("../src/cavi.jl");

@testset "cavi.jl" begin
    
    @testset "compApproxMarginals!(approxMarginals, traces; caviCounter, spatialScheme)" begin
        
        nEpoch = 1;
        epochSize = 1;
        M = 4;

        approxMarginals = Vector{Distribution}(undef, M+3);
        traces = Dict(
            :muMean => zeros(M, nEpoch*epochSize),
            :phiMean => zeros(M, nEpoch*epochSize),
            :xiMean => zeros(nEpoch*epochSize),
            :cellVar => fill(1.0, M, nEpoch, 4),
            :xiVar => ones(M),
            :kappaUparams => [1; 2;;],
            :kappaVparams => [1; 2;;],
        )
        caviCounter = Dict(
            :iter => 1,
            :epoch => 1,
            :numCell => 1,
        )
        spatialScheme = Dict(
            :M => M,
            :Fmu => iGMRF(2, 2, 1, 10.0),
            :Fphi => iGMRF(2, 2, 1, 100.0),
            :data => [
                [1.0, 1.0],
                [1.0, 1.0],
                [1.0, 1.0],
                [1.0, 1.0],
            ],
        );

        compApproxMarginals!(approxMarginals, traces, caviCounter=caviCounter, spatialScheme=spatialScheme);

        @test approxMarginals[M+3] == Gamma(1, .5);
    end


    @testset "estimateKappa(variant; traces, caviCounter)" begin
        
        traces = Dict(
            :kappaUparams => [0 1; 0 2],
        );
        caviCounter = Dict(
            :iter => 2,
        );

        @test estimateKappa("u", traces=traces, caviCounter=caviCounter) ≈ .5;
    end
end