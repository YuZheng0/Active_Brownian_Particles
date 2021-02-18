import Statistics
#using PyPlot

const evolveRange = 1
T = 0.000001

function calculateInnerProduct(grids, distanceForCorr, innerProduct)
    for i = 1:NUMGRID
        for j = 1:NUMGRID
            for i2 = 1:NUMGRID
                for j2 = 1:NUMGRID
                    distanceForCorr[i,j,i2,j2] = boundaryDistance(grids[i,j],grids[i2,j2])
                    innerProduct[i,j,i2,j2] =  grids[i,j].direction[1] * grids[i2,j2].direction[1]+ grids[i,j].direction[2]*grids[i2,j2].direction[2]
                end
            end      
        end
    end  
end 

function correlationValue(grids, r, sizeBin,inner,distanceForCorr,innerProduct)
    deltaR = sizeBin * r
    for i = 1:NUMGRID
        for j = 1:NUMGRID
            for i2 = 1:NUMGRID
                for j2 = 1:NUMGRID
                    if distanceForCorr[i,j,i2,j2] <= (r + deltaR / 2) && distanceForCorr[i,j,i2,j2] >= (r - deltaR / 2)
                    append!(inner, innerProduct[i,j,i2,j2])
                    end
                end
            end
        end
    end
    return Statistics.mean(inner)
end

function correlationFunction(grids,sizeBin,rStepCoeff,correlationX,correlationY,inner,distanceForCorr,innerProduct)
    initr = LENGTH / NUMGRID
    rstep = initr * rStepCoeff
    for r in initr:rstep:LENGTH/2
        append!(correlationX, r)
        append!(correlationY, correlationValue(grids,r,sizeBin,inner,distanceForCorr,innerProduct))
    end
    #scatter(correlationX,correlationY)
    #savefig("1.png")
end

function getEnergy(grids,sizeBin,rStepCoeff,alpha,correlationX,correlationY,inner,distanceForCorr,innerProduct)
    correlationFunction(grids,sizeBin,rStepCoeff,correlationX,correlationY,inner,distanceForCorr,innerProduct)
    theoretical = exp.(alpha * correlationX)
    
    abs2value = abs2.(theoretical - correlationY)
    energy = sum(abs2value)

    return energy
end

function evolve(grids, distanceForCorr, innerProduct,rotateAngle, i,j)
    grids[i,j].direction = rotateAtoB(grids[i, j].direction, rotateAngle)
    for i2 = 1:NUMGRID
        for j2 = 1:NUMGRID
            distanceForCorr[i,j,i2,j2] = boundaryDistance(grids[i,j],grids[i2,j2])
            distanceForCorr[i2,j2,i,j] = boundaryDistance(grids[i,j],grids[i2,j2])
            innerProduct[i,j,i2,j2] =  grids[i,j].direction[1] * grids[i2,j2].direction[1]+ grids[i,j].direction[2]*grids[i2,j2].direction[2]
            innerProduct[i2,j2,i,j] =  grids[i,j].direction[1] * grids[i2,j2].direction[1]+ grids[i,j].direction[2]*grids[i2,j2].direction[2]
        end
    end    
end

function isAccepted(energy,newEnergy)
    if newEnergy > energy
        deltaEnergy = newEnergy - energy 
        #println("deltaEnergy: ", deltaEnergy) 
        relativePossibility = exp(-deltaEnergy/T)  
        #println("relativePossibility: ", relativePossibility)
        test = rand(uniform)
        #println("test: ",test)
	    if test > relativePossibility
            return false
        end
    end
    return true
end

function generateFiber(grids,sizeBin,rStepCoeff,alpha,numOfRun)

    distanceForCorr = Array{Float64}(undef, NUMGRID,NUMGRID,NUMGRID,NUMGRID)
    innerProduct = Array{Float64}(undef, NUMGRID,NUMGRID,NUMGRID,NUMGRID)
    
    calculateInnerProduct(grids, distanceForCorr, innerProduct)
    
    for run = 1:numOfRun
        
        correlationX = []
        correlationY = []
        inner = []

        energy = getEnergy(grids,sizeBin,rStepCoeff,alpha,correlationX,correlationY,inner,distanceForCorr,innerProduct)
        #scatter(correlationX, correlationY, color="b")
        #savefig("1.png") 

        if run ==1
            println("initial energy:", energy)
        end

        rotateAngle = evolveRange * (rand(uniform)-0.5) 
        i = rand(1:NUMGRID)
        j = rand(1:NUMGRID)

        evolve(grids, distanceForCorr, innerProduct,rotateAngle,i,j)
        
        correlationX = []
        correlationY = []
        inner = []

        newEnergy = getEnergy(grids,sizeBin,rStepCoeff,alpha,correlationX,correlationY,inner,distanceForCorr,innerProduct)

        if !(isAccepted(energy,newEnergy))
            evolve(grids, distanceForCorr, innerProduct,-rotateAngle,i,j)
            #println("energy: ", energy)
        else
            #println("newEnergy: ",newEnergy)
           
        end

        if run == numOfRun
            correlationX = []
            correlationY = []
            inner = []
            println("final energy: ", getEnergy(grids,sizeBin,rStepCoeff,alpha,correlationX,correlationY,inner,distanceForCorr,innerProduct) )
        end        
    end
    
    # correlationX = []
    # correlationY = []
    # inner = []
    # correlationFunction(grids,sizeBin,rStepCoeff,correlationX,correlationY,inner,distanceForCorr,innerProduct)
    # scatter(correlationX, exp.(alpha * correlationX), color="r")
    # scatter(correlationX, correlationY, color="b")
    # savefig("1.png") 

end


