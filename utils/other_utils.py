import numpy as np, time, math
import matplotlib.pyplot as plt

def read_cbin(filename, datatype=np.float32, order='C', dimensions=3):
    f = open(filename)
    header = np.fromfile(f, count=dimensions, dtype='int32')
    data = np.fromfile(f, dtype=datatype, count=np.prod(header))
    data = data.reshape(header, order=order)
    f.close()
    return data

def save_cbin(filename, data, datatype=np.float32, order='C'):
    f = open(filename, 'wb')
    mesh = np.array(np.shape(data)).astype('int32')
    mesh.tofile(f)
    np.ravel(a=data, order=order).astype(datatype).tofile(f)
    f.close()

class TimerError(Exception): 
    """A custom exception used to report errors in use of Timer class""" 

class Timer: 
    def __init__(self): 
        self._start_time = None
        self._prevlap_time = None
        self._summary = '\n--- TIMER SUMMARY ---\n'

    def _display(self, chrn):
        if(chrn >= 60): 
            secs = chrn % 60 
            mins = chrn // 60 
            if(mins >= 60): 
                mins = mins % 60
                hrs = chrn // 60 // 60 
                display = '%d hrs %d min %.2f sec'  %(hrs, mins, secs)
            else: 
                display = '%d mins %.2f sec'  %(mins, secs)
        else: 
            display = '%.2f sec'  %chrn
        return display

    def start(self): 
        """Start a new timer""" 
        if(self._start_time != None): 
            raise TimerError(f"Timer is running. Use .stop() to stop it") 
        self._start_time = time.perf_counter()

    def lap(self, mess=None): 
        """Stop the timer, and report the elapsed time""" 
        if(self._start_time == None): 
            raise TimerError(f"Timer is not running. Use .start() to start it") 
        lap_time = time.perf_counter()
        if(self._prevlap_time != None):
            elapsed_time = lap_time - self._prevlap_time
        else:
            elapsed_time = lap_time - self._start_time
        self._prevlap_time = lap_time
        mess = ' - '+str(mess) if mess!=None else ''
        text_lap = "Lap time: %s %s" %(self._display(elapsed_time), mess)
        self._summary += ' ' + text_lap+'\n'
        #print(text_lap) 

    def stop(self, mess=''): 
        """Stop the timer, and report the elapsed time""" 
        if(self._start_time == None): 
            raise TimerError(f"Timer is not running. Use .start() to start it")
        time_stop = time.perf_counter()
        if(self._prevlap_time != None):
            elapsed_time = time_stop - self._prevlap_time
            text_lap = "Lap time: %s - final lap" %(self._display(elapsed_time))
            self._summary += ' ' + text_lap
            print(self._summary)
            #print(text_lap)
        elapsed_time = time_stop - self._start_time
        mess = ' - '+str(mess) if mess!='' else mess
        print("Elapsed time: %s %s" %(self._display(elapsed_time), mess)) 
        self._start_time = None


def PercentContours(x, y, bins=None, colour='green', style=[':', '--', '-'], perc_arr=[0.99, 0.95, 0.68], lw=3):
    if(type(bins) == int):
        hist, xedges, yedges = np.histogram2d(x, y, bins=bins)
    elif(bins == 'log'):
        x_edges = np.logspace(np.log10(np.min(x)), np.log10(np.max(x)), 100)
        y_edges = np.logspace(np.log10(np.min(y)), np.log10(np.max(y)), 100)
        hist, xedges, yedges = np.histogram2d(x, y, bins=(x_edges, y_edges))
    elif(bins == 'lin'):
        x_edges = np.linspace(np.min(x), np.max(x), 100)
        y_edges = np.linspace(np.min(y), np.max(y), 100)
        hist, xedges, yedges = np.histogram2d(x, y, bins=(x_edges, y_edges))

    sort_hist = np.sort(hist.flatten())[::-1]
    perc = (np.array(perc_arr)*np.sum(sort_hist)).astype(int)
    levels = np.zeros_like(perc)
    
    j = -1
    for i, val in enumerate(sort_hist):
        if(np.sum(sort_hist[:i]) >= perc[j]):
            levels[j] = val
            if(j == -len(perc)):
                break
            j -= 1 
    #c = plt.contour(hist.T, extent=[xedges.min(), xedges.max(), yedges.min(), yedges.max()], levels=levels, colors=colour, linestyles=style, linewidths=lw)
    x_plot, y_plot = 0.5*(x_edges[1:]+x_edges[:-1]), 0.5*(y_edges[1:]+y_edges[:-1])

    c = plt.contour(x_plot, y_plot, hist.T, levels=levels, colors=colour, linestyles=style, linewidths=lw)
    
    c.levels = np.array(perc_arr)*100.
    plt.clabel(c, c.levels, inline=True,inline_spacing=10, fmt='%d%%', fontsize=16)
    #plt.draw()
