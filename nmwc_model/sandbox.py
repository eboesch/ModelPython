import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from nmwc_model.readsim import readsim

"""
out = np.load('output.npz')
print(out['z'][34])

print(np.abs(-5))

#for i in range(10,5,(-1)):
#    print(i)
"""
nx = 100
nb = 2
nxb = nx + 2 * nb  # x range of unstaggered variable
#print(nxb)
xl = 500000.0  # domain size  [m]
nx = 100  # number of grid points in horizontal direction
dx = xl / nx  # horizontal resolution [m]
#print(dx)

topomx = 500  # mountain height [m]
topowd = 50000  # mountain half width [m]
topotim = 1800  # mountain growth time [s]


topo = np.zeros((nxb, 1))


x = np.arange(0, nxb, dtype=np.float64)
#print(x)
#topo[1:-1, 0] = x[1:-1]
#print(topo)

x0 = (nxb - 1) / 2.0 + 1 #get the middle of the x axis
#print(x0)
x = (x + 1 - x0) * dx # translate x by x0 and scale by dx
print(x)
toponf = topomx * np.exp(-(x / float(topowd)) ** 2) #make gaussian curve
print(toponf)


# overwrites all but first and last entry of topo
topo[1:-1, 0] = toponf[1:-1] + 0.25 * (
    toponf[0:-2] - 2.0 * toponf[1:-1] + toponf[2:] # weighted average of 2*current point + prev point + next point -> smooths it out?
)





# plot
op = arg_parser()
# get command line arguments
args = op.parse_args()
# ds = Dataset(args.filename[0], 'r')
timestep = args.time
varnames = args.varname.split(",")
var = readsim(args.filename[0], varnames)

#plot_figure(varnames, var, timestep, True)
plt.sca(ax)
ax.cla()

ax.xaxis.set_minor_locator(MultipleLocator(50))
ax.yaxis.set_minor_locator(MultipleLocator(1))

ax.set_xlabel("x [km]")
ax.set_ylabel("Height [km]")

    # Add theta
if args.tlim[0] == 0:
    tlim_min = var.th00 + var.dth / 2
else:
    tlim_min = args.tlim[0]

clev = np.arange(tlim_min, args.tlim[1], args.tci)

plt.contour(
    var.xp[:, :], var.zp[timestep, :, :],
        var.theta[:, :], clev, colors="grey", linewidths=1
)

# Add topography
plt.plot(var.xp[0, :], var.topo, "-k")
plt.ylim(args.zlim)

"""
    for varname in varnames:
        # Determine range for values and ticks
        valRange = np.arange(
            pd[varname]["clev"][0],
            pd[varname]["clev"][-1] + pd[varname]["ci"],
            pd[varname]["ci"],
        )
        ticks = np.arange(
            pd[varname]["clev"][0],
            pd[varname]["clev"][-1] + pd[varname]["ci"],
            pd[varname]["ci"],
        )

        vmin = valRange[0]
        vmax = valRange[-1]
        if varname == "horizontal_velocity":
            valRange = np.arange(
                pd[varname]["clev"][0] - 0.5 * pd[varname]["ci"],
                pd[varname]["clev"][-1] + 1.5 * pd[varname]["ci"],
                pd[varname]["ci"],
            )
            distUpMid = pd[varname]["clev"][-1] + 0.5 * args.vci - var.u00
            distMidDown = var.u00 - pd[varname]["clev"][0] - 0.5 * args.vci
            maxDist = max(distUpMid, distMidDown)
            vmin = var.u00 - maxDist
            vmax = var.u00 + maxDist
            

        # Plot
        cs = ax.contourf(
            var.xp[:, :],
            var.zp[timestep, :, :],
            pd[varname]["scale"] * var[varname][timestep, :, :],
            valRange,
            vmin=vmin,
            vmax=vmax,
            cmap=pd[varname]["cmap"],
        )

        # Add a colorbar if needed
        if plot_cbar:
            cb = plt.colorbar(cs, ticks=ticks, spacing="uniform")

        if varname == "specific_rain_water_content":
            tpi = 0.1
            vMinInt = args.totpreclim[0]
            vMaxInt = args.totpreclim[1]

            # If no upper limit is given, it will be set automatically
            # according to the values in the data.
            if np.isnan(vMaxInt):
                maxTp = np.nanmax(var.accumulated_precipitation[:, :])
                tpDiff = int((maxTp - vMinInt) / tpi + 1.0)
                vMaxInt = vMinInt + tpDiff * tpi

            ax2.cla()
            ax2.set_ylabel("Acum. Rain [mm]")
            #             ax2.set_ylim(args.totpreclim)
            ax2.set_ylim(vMinInt, vMaxInt)
            cs = ax2.plot(
                var.xp[0, :], var.accumulated_precipitation[timestep, :], "b-"
            )
            """


# plt.show()