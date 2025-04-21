using Plots, ColorSchemes
plotlyjs()

function plot_fun(f; points=nothing, limits=nothing, params=nothing)
    limits_dict = Dict("ackley"=> (-5, 5, -5, 5),
                   "bukin"=> (-15, -5, -3, 3),
                   "cross_in_tray"=> (-5, 5, -5, 5),
                   "drop_wave"=> (-5.2, 5.2, -5.2, 5.2),
                   "eggholder"=> (-520, 520, -520, 520),
                   "griewank"=> (-5, 5, -5, 5),
                   "holder_table"=> (-10, 10, -10, 10),
                   "langermann"=> (0,10, 0, 10),
                   "levy"=> (-10, 10, -10, 10),
                   "bochavesky"=> (-100, 100, -100, 100),
                   "perm_zero_d_beta"=> (-2, 2, -2, 2),
                   "rot_hyper_ellipsoid"=> (-100, 100, -100, 100),
                   "sphere"=> (-100, 100, -100, 100),
                   "sum_powers"=> (-1, 1, -1, 1),
                   "sum_squares"=> (-10, 10, -10, 10),
                   "trid"=> (-4, 4, -4, 4),
                   "booth"=> (-10, 10, -10, 10),
                   "matyas"=> (-10, 10, -10, 10),
                   "mccormick"=> (-1.5, 4, -3, 4),
                   "power_sum"=> (-2, 2, -2, 2),
                   "zakharov"=> (-5, 10, -5, 10),
                   "three_hump_camel"=> (-5, 5, -5, 5),
                   "six_hump_camel"=> (-1.5, 1.5, -1.5, 1.5),
                   "dixon_price"=> (-10, 10, -10, 10),
                   "rosenbrock"=> (-2.048, 2.048, -2.048, 2.048),
                   "de_jong_5"=> (-50, 50, -50, 50),
                   "easom"=> (-20, 20, -20, 20),
                   "michalewicz"=> (0, 4, 0, 4),
                   "beale"=> (-4.5, 4.5, -4.5, 4.5),
                   "branin"=> (-5, 15, -5, 15),
                   "goldstein_price"=> (-2, 2, -2, 2),
                   "perm_d_beta"=> (-3, 3, -3, 3),
                   "styblinski_tang"=> (-5, 5, -5, 5),
                   "parsopoulos"=> (-5, 5, -5, 5)
                    )

    if limits === nothing
        limits = limits_dict[string(Symbol(f))]
    end
    x = range(limits[1], limits[2], length=1000)
    y = range(limits[3], limits[4], length=1000)
    
    if params===nothing
        display(surface(x,y,f,c=cgrad(:vik)))
    else
        F(x,y) = f(x,y,params...)
        display(surface(x,y,F,c=cgrad(:vik)))
    end
    if points !== nothing
        scatter!(points[:,1],points[:,2],points[:,2])
    end
end


# -------------------------------------MANY LOCAL MINIMA-------------------------------------------------
# -------------------------------------------------------------------------------------------------------
"""
    ackley function
    minimiser = (0,0)
"""
function ackley(x::Vector{Float64}; a=20., b=0.2, c=2pi)
    d = length(x)
    sum_1 = sqrt(1/d*sum(x.^2))
    sum_2 = sum(cos.(c*x))
    return -a*exp(-b*sum_1) - exp(1/d*sum_2) + a + exp(1)
end
ackley(x::Number,y::Number) = ackley([x,y])


"""
    bukin function
    minimiser: x = (-10,1)
"""
bukin(x,y) = sqrt(abs(y-0.01*x^2))+0.01*abs(x+10)

"""
    cross_in_tray function. 
    minimisers: x = (1.3491, -1.3491), (1.3491, 1.3491), (-1.3491, 1.3491), (-1.3491, -1.3491)
"""
cross_in_tray(x,y) = -1e-4*((abs(sin(x)sin(y)exp(abs(100-sqrt(x^2+xy^2)/pi)))+1)^0.1)

"""
    drop_wave function
    minimiser: x = (0,0)
"""
drop_wave(x,y) = -(1 + cos(12*sqrt(x^2+y^2)))/(0.5*(x^2 + y^2)+2)
"""
    minimiser: x = (512, 404.2319)
"""

eggholder(x,y) = -(x+47)sin(sqrt(abs(y + x/2 + 47)))-xsin(sqrt(abs(x-(y+47))))
"""
    minimiser : x = 0
"""

griewank(x::Vector{Float64}) = sum(x.^2)/4000 - prod(cos.(x)./sqrt.(1:length(x))) + 1
griewank(x::Float64,y::Float64) = griewank([x,y])
"""
    minimisers : x = (8.05502, 9.66459), (8.05502, -9.66459), (-8.05502, 9.66459), (-8.05502, -9.66459)
"""

holder_table(x,y) = abs(sin(x)cos(y)*exp(abs(1-sqrt(x^2+y^2)/pi)))
"""
    minimiser: x= (2.00299219, 1.006096)
"""

function langermann(x::Vector{Float64};c=[1, 2, 5, 2, 3], a=[3 5 2 1 7;5 2 1 4 9]')
    m,d = size(a)
    return sum(c[i]exp(-1/pi * sum((x[j] - a[i, j])^2 for j in 1:d)) *
               cos(pi*sum((x[j] - a[i, j])^2 for j in 1:d)) for i in 1:m)
end
langermann(x,y) = langerann([x,y])
"""
minimiser: x = (1,...,1)
"""

function levy(x::Vector{Float64})
    w(y) = 1 + (y-1)/4
    if length(x)>2
        s = sum((w.(x[i])-1).^2 .*(1+10sin(pi*w.(x[i])+1).^2) for i in 2:length(x)-1)
    else 
        s = 0
    end
    return sin(pi*w(x[1]))^2 + s + (w(x[end])-1)^2*(1+sin(2*pi*w(x[end]))^2)
end
levy(x::Float64,y::Float64) = levy([x,y]) 

# -------------------------------------BOWL-SHAPED-------------------------------------------------------
# -------------------------------------------------------------------------------------------------------

"""
    bochavesky functions
    minimiser: (x,y) = (0,0)
    n is the number of the bochavesky function
"""
function bochavesky(x,y; n=1)
    if n == 1
        return x^2+2(y^2)-0.3cos(3pi*x)-0.4cos(4pi*y)+0.7
    elseif n == 2
        return x^2+2(y^2)-0.3cos(3pi*x)*cos(4pi*y)+0.3
    elseif n == 3
        return x^2+2(y^2)-0.3cos(3pi*x+4pi*y)+0.3
    end
end

"""
    perm_zero_d_beta function
    minimiser: (1, 1/2, ..., 1/d)
"""
function perm_zero_d_β(x::Vector{Float64};β=1)
    return sum(sum((j+β)*(x[j]^i-1/(j^i)) for j in 1:length(x))^2 for i in 1:length(x))
end
perm_zero_d_β(x::Number,y::Number) = perm_zero_d_β([x,y])

"""
    rot_hyper_ellipsoid function
    minimiser: x = 0
"""
rot_hyper_ellipsoid(x::Vector{Float64}) = sum(sum(x[j]^2 for j in 1:i) for i in 1:length(x))
rot_hyper_ellipsoid(x::Number,y::Number) = rot_hyper_ellipsoid([x,y])


"""
    sphere
    minimiser: x = 0
"""
sphere(x::Vector{Float64}) = sum(x.^2)
sphere(x::Number,y::Number) = x^2+y^2

"""
    su_powers function
    minimiser: x = 0
"""
sum_powers(x::Vector{Float64}) = sum(abs(x[i])^(i+1) for i in 1:length(x))
sum_powers(x::Number) = x^2+abs(y)^3

"""
    sum_squares
    minimiser: x = 0
"""
sum_squares(x::Vector{Float64}) = (1:length(x))⋅(x.^2)
sum_squares(x::Number,y::Number) = x^2+2y^2

"""
    trid function
    minimiser: xᵢ   = i(d+1-i) ∀ i=1,2,...,d
"""
trid(x::Vector{Float64}) = sum((x.-1).^2) - sum(x[2:end].*x[1:end-1])
trid(x::Number,y::Number) = trid([x,y])

# -------------------------------------PLATE-SHAPED------------------------------------------------------
# -------------------------------------------------------------------------------------------------------
"""
    booth function
    minimiser: x= (1,3)
"""
booth(x,y) = (x + 2y - 7)^2 + (2x + y - 5)^2

""" 
    matyas function
    minimiser: x= (0,0)
"""
matyas(x,y) = 0.26(x^2 + y^2) - 0.48x*y

"""
    mccormick function
    minimiser: x= (-0.54719, -1.54719)
"""
mccormick(x,y) = sin(x + y) + (x - y)^2 - 1.5x + 2.5y + 1

"""
    power_sum function
"""  
power_sum(x::Vector{Float64}, b=(8, 18)) = sum((sum(x[j]^(i+1) for j in 1:length(x))-b[i])^2 for i in 1:length(x))
power_sum(x::Number,y::Number) = power_sum([x,y])

"""
    zakharov function
    minimiser: x = 0
"""
zakharov(x::Vector{Float64}) = sum(x.^2) + sum(0.5*(1:length(x)).*x)^2 + sum(0.5*(1:length(x)).*x)^4
zakharon(x::Number,y::Number) = zakharov([x,y])

# -------------------------------------VALLEY-SHAPED-----------------------------------------------------
# -------------------------------------------------------------------------------------------------------
"""
    three_hump_cael
    global minimiser: x= (0,0)
    Has three local minima
"""
three_hump_camel(x,y) = 2x^2 - 1.05x^4 + (x^6)/6 + x*y + y^2

"""
    six_hump_camel
    global minimisers : x = (0.0898, -0.7126)  and  x = (-0.0898, 0.7126)
    Has six local minima
"""
six_hump_camel(x,y) = (4 - 2.1x^2 + (x^4)/3)*(x^2) + x*y + (-4 + 4y^2)*(y^2)

"""
    dixon_price function
    minimiser (for d=2): x = (1,1/sqrt(2))
"""
dixon_price(x::Vector{Float64}) = sum(i*(2*x[i]^2-x[i-1])^2 for i in 2:length(x)) + (x[1] - 1)^2
dixon_price(x::Number,y::Number) = dixon_price([x,y])

"""
    rosenbrock function
    minimiser : x = (1,..., 1)
"""
rosenbrock(x::Vector{Float64}) = sum(100(x[i+1]-x[i]^2)^2+(x[i]-1)^2 for i in 1:length(x)-1)
rosenbrock(x::Number,y::Number) = rosenbrock([x,y])

# -------------------------------------STEEP RIDGES/DROPS-----------------------------------------
# ------------------------------------------------------------------------------------------------
function de_jong_5(x)
    r1 = repeat([-32, -16, 0, 16, 32],5)
    r2 = [-32*ones(5);-16*ones(5);zeros(5);16*ones(5);32*ones(5)]
    a = [r1 r2]
    return 1/(0.002 + sum(1/(i + (x[1]-a[i,1])^6 + (x[2]-a[i,2])^6) for i in 1:25))
end

"""
    easom function
    minimiser : x = (pi,pi)
"""
easom(x,y) = -cos(x)cos(y)exp(-(x-pi)^2-(y-pi)^2)

"""
    michalewicz function
    m defines steepness of valleys and ridges
    Has d! local minima
"""
michalewicz(x::Vector{Float64};m=10) = -sum(sin(x[i])sin(i*(x[i]^2)/pi)^(2m) for i in 1:length(x))
michalewicz(x::Number,y::Number) = michalewicz([x,y])

# -------------------------------------OTHERS------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------
"""
    beale function
    minimiser : x = (3,0.5)
"""
beale(x,y) = (1.5-x+x*y)^2 + (2.25-x+x*y^2)^2 + (2.625-x+x*y^3)^2

"""
    branin function
    minimisers : x = (-pi,12.275) , (pi, 2.275), (9.42478, 2.475)
"""
branin(x,y; a=1, b=5.1/4pi^2, c=5/pi, r=6, s=10, t=1/8pi) = 
        a*(y - b*x+c*x - r)^2 + s*(1-t)*cos(x) + s

"""
    colville function
    minimiser : x = (1,1,1,1)
"""
colville(x) = 100(x[1]^2-x[2])^2 + (x[1]-1)^2 + (x[2]-1)^2 + 90(x[3]^2-x[4])^2 + 10.1((x[2]-1)^2 + (x[4]-1)^2) + 19.8(x[2]-1)*(x[4]-1)


"""
    goldstein_price
    minimiser : x = (0,-1)
"""
goldstein_price(x,y) = 1 + (x+y+1)^2*(19-14x+3x^2-14y+6x*y+3y^2)+ 30 + (2x - 3y)^2*(18- 32x + 12y^2 + 48y - 36x*y + 27y^2)

"""
    perm_d_beta
    minimiser : x = (1,2,...,d)
"""
perm_d_beta(x::Vector{Float64}; β=1) = sum(sum((j^i+β)*((x[j]/j)^i-1) for j in 1:length(x))^2 for i in 1:length(x))
perm_d_beta(x::Number,y::Number) = perm_d_beta([x,y])

"""
    minimiser : x = (-2.903534,...,-2.903534)
"""
styblinski_tang(x::Vector{Float64}) = 0.5sum(x.^4 - 16x.^2 + 5x)
styblinski_tang(x::Number,y::Number) = styblinski_tang([x,y])

"""
    parsopoulos
    ∞ global minimisers: x = (k*(pi/2), l*pi) for k=1,2,3,... and l= 0, 1, 2, ...
"""
parsopoulos(x,y) = cos(x)^2 + sin(y)^2


# plot_fun(langermann, points=[np.array([0.5, 0.32, levy(np.array([0.5, 0.32]))])])
