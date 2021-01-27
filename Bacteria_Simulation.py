import warnings
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm

'''This program provides a basic model for the behaviour of bacteria in proximity to a 'food source' (located at the origin). At each timestep,
the bacteria decide to either continue in their current direction or undergo a random direction change; the former is chosen when the bacteria detects
that it has moved closer to the food source, and the latter chosen when the bacteria detects it is getting further from the food source. The result is a
characteristic 'biased' random walk, with the bacteria getting closer and closer to the origin.'''

warnings.filterwarnings("ignore")

plt.rcParams.update({"text.usetex": True,"font.family": "sans-serif","font.sans-serif": ["Helvetica"]})

parser=argparse.ArgumentParser()
parser.add_argument('-dt',type=float,default=0.1,help='Step size in seconds.')
parser.add_argument('-k',type=float,default=0.2,help='Walk penalty. walk_probability = probability_function(dt/1 + k * de / dt), so larger k => more sensitive to changes in energy.')
parser.add_argument('-num_bacteria',type=int,default=25,help='Number of bacteria to simulate.')
parser.add_argument('-timesteps',type=int,default=1000,help='Number of timesteps for simulation.')
parser.add_argument('-initial_pos',type=float,nargs=2,default=None,help='Starting position for bacteria. Default is to choose starting positions randomly.')
parser.add_argument('-initial_speed',type=float,default=1,help='Initial velocity magnitude.')
parser.add_argument('-lag',type=int,default=9,help='Lag.')
parser.add_argument('-x_extent',type=int,default=50,help='x_extent')
parser.add_argument('-y_extent',type=int,default=50,help='y_extent')
parser.add_argument('-probability_function',type=str,default='exponential',help='Prob func.')
args=parser.parse_args()

dt = args.dt
k = args.k
num_bacteria = args.num_bacteria
timesteps = args.timesteps
initial_pos = args.initial_pos
initial_speed = args.initial_speed
lag = args.lag
x_extent = args.x_extent
y_extent = args.y_extent
probability_function = args.probability_function

randomise_starting_positions = False
if not initial_pos:
    randomise_starting_positions = True

def energy_density(r): # energy density function
    x, y = r # unpack position vector
    energy_density = 4000 - (x**2 + y**2)
    return energy_density

def walk_probability(time_constant):
    '''Calculate probability that a walk event takes place (i.e. NOT a tumble), using the user's specified probability function.'''
    if probability_function == 'exponential':
        walk_prob = np.exp(-dt/time_constant)
    elif probability_function == 'gaussian':
        walk_prob = np.exp( -(dt/time_constant)**2 )
    elif probability_function == 'fractional':
        walk_prob = 1 / ( 1 + (dt/time_constant) )
    elif probability_function == 'lorentzian':
        walk_prob = 1 / ( 1 + (dt/time_constant)**2 )
    else:
        print('ERROR: Specified probability function not recognised.')
        exit()
    return walk_prob

def random_walk(initial_pos):
    r_array = np.zeros((timesteps, 2)) # array of zeros of shape (timesteps, 2), i.e. to hold timesteps occurences of (x,y) coordinate tuples
    r = initial_pos
    angle_rand = np.random.random() * 2 * np.pi # generates a random angle between 0 and 2pi, here used to assign initial velocity randomly
    initial_vel = (initial_speed*np.cos(angle_rand), initial_speed*np.sin(angle_rand)) # Random starting velociy of magnitude initial_speed
    vel = initial_vel # Initialise vel
    shift = (lag+1) * [0] # shift register array to hold current and historical values of energy density function
    for i in range (timesteps):
        x, y = r # unpack position
        vx, vy = vel # unpack velocity
        r_array[i] = r
        eNew = energy_density(r)
        shift.append(eNew)
        shift = shift[-(lag+1):]
        de = shift[-1] - shift[0] # find difference between current and value from lag timesteps ago
        time_constant = 1 + k * de / dt
        if time_constant < 0.1:
            time_constant = 0.1 # prevents negative half lives by forcing time_constant to be 0.1 if below threshold value of 0.1
        walk_prob = walk_probability(time_constant) # probability that a walk actually takes place, or rather that a tumble does NOT take place   
        if walk_prob > np.random.random() : # condition for walk
            xnew, ynew = x + vx * dt, y + vy * dt # walk in direction of vel
            r = (xnew,ynew)
        else: # else tumble
            angle_rand = np.random.random() * 2 * np.pi # generates a random angle between 0 and pi, used to generate new velocity below
            vel = (np.cos(angle_rand)*vx - np.sin(angle_rand)*vy,
                   np.sin(angle_rand)*vx + np.cos(angle_rand)*vy) # new vel vect determined from randomly generated angle by applying a rotation (mag. is unaffected)
    return r_array

x0, x1, = -x_extent, x_extent
y0, y1 = -y_extent, y_extent
N_POINTS = 50
dx = (x1-x0)/N_POINTS
dy = (y1-y0)/N_POINTS

def plot_trajectories(initial_pos): # plotting function, called upon code start to produce the three plots
    plt.figure()
    plt.subplots_adjust(hspace=0.,wspace=0.)
    y_axis = np.arange(y0,y1,dy)
    x_axis = np.arange(x0,x1,dx)
    dat = np.zeros((len(y_axis), len(x_axis)))
    for iy, y in enumerate (y_axis):
        for ix, x in enumerate (x_axis):
            dat[iy, ix] = energy_density((x_axis[ix], y_axis[iy]))
    plt.subplot(2,2,1)
    plt.title('Complete Bacteria Trajectories', fontsize=14)
    plt.xlabel(r'$x$', fontsize=15), plt.ylabel(r'$y$', fontsize=15) # xlabel
    im = plt.imshow(dat, extent=(x0, x1, y0, y1), origin='lower', cmap=matplotlib.cm.afmhot, aspect='auto')
    X, Y = np.meshgrid(x_axis, y_axis)
    contours1 = plt.contour(X, Y, dat)
    plt.clabel(contours1, inline=1, fontsize=8)
    plt.subplot(2,2,2)
    plt.title('Start and End Points', fontsize=14)
    plt.xlabel(r'$x$', fontsize=15), plt.ylabel(r'$y$', fontsize=15) # xlabel
    im = plt.imshow(dat, extent=(x0, x1, y0, y1), origin='lower', cmap=matplotlib.cm.afmhot, aspect='auto')
    contours2 = plt.contour(X, Y, dat)
    plt.clabel(contours2, inline=1, fontsize=8)
    plt.colorbar(im, orientation='vertical', label='Energy Density $(x,y)$')
    xs_array = num_bacteria * [0]
    ys_array = num_bacteria * [0]
    displ_start_squared_array = num_bacteria * [0] # will hold displacement^2 from starting point values
    displ_origin_squared_array = num_bacteria * [0] # will hold displacement^2 from the origin (0,0) values
    time_array = np.arange(0,timesteps*dt,dt)
    for j in range(num_bacteria): # a loop runs for every bacterium
            if randomise_starting_positions:
                initial_pos = [x_extent*(np.random.random()-0.5),y_extent*(np.random.random()-0.5)]
            xs_array[j] = random_walk(initial_pos)[:,0]
            ys_array[j] = random_walk(initial_pos)[:,1]
            time_array[j] = j * 0.1 # stores accumulated time in seconds
            displ_start_squared_array[j] = (xs_array[j]-xs_array[j][0])**2 + (ys_array[j]-ys_array[j][0])**2 # displacement from starting point
            MSD_start = np.average(displ_start_squared_array, axis=0) # averages over all bacteria at a given time to give mean squared displacement
            displ_origin_squared_array[j] = (xs_array[j]-0)**2 + (ys_array[j]-0)**2 # displacement from origin (where food source is located)
            MSD_origin = np.average(displ_origin_squared_array, axis=0) # averages over all bacteria at a given time to give mean squared displacement
            xinit_fin = (xs_array[j][0], xs_array[j][timesteps-1]) # first and final x coordinates, used to produce the second plot
            yinit_fin = (ys_array[j][0], ys_array[j][timesteps-1]) # first and final y coordinates, used to produce the second plot
            plt.subplot(2,2,1)
            plt.ylim(y0, y1), plt.xlim(x0, x1)
            plt.plot(xs_array[j], ys_array[j])
            plt.subplot(2,2,2)
            plt.ylim(y0, y1), plt.xlim(x0, x1)
            plt.plot(xinit_fin,yinit_fin, marker='o')
    plt.subplot(2, 1, 2)
    plt.title('Squared Displacements Vs Time', fontsize=14)
    plt.xlabel('Elapsed Time (seconds)', fontsize=14), plt.ylabel('MSD (microns)', fontsize=14)
    plt.plot(time_array, MSD_start, label='MSD from Starting Point', color='m') # mean squared displacement from starting point Vs time
    plt.plot(time_array, MSD_origin, label='MSD from Food Source') # mean squared displacement from the origin (0,0), which is the point of greatest energy density
    plt.legend(loc=7) # gives the plot a legend
    plt.tight_layout() # prevents overlapping of plots/axis labels
    plt.show()

plot_trajectories(initial_pos)


