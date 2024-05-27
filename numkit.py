from numpy import array, arange, zeros, zeros_like, sum, linspace, abs


def diff_2p(f,x_0,h,*args):
    return (f(x_0+h,*args)-f(x_0,*args))/h


def diff_4p(f,x_0,h,*args):
    return (-f(x_0 + 2 * h,*args) + 8 * f(x_0 + h,*args) - 8 * f(x_0 - h,*args) + f(x_0 - 2 * h,*args)) / (12 * h)


def diff_6p(f,x_0,h,*args):
    return (-f(x_0-3*h,*args)+9*f(x_0-2*h,*args)-45*f(x_0-h,*args)+45*f(x_0+h,*args)-9*f(x_0+2*h,*args)+f(x_0+3*h,*args))/(60*h)


def int_riemann(f, a, b, n, *args):
    """
    Berechne das Integral einer Funktion mit Hilfe der linken Riemann-Summe

    Parameters:
        f: Zu integrierende Funktion
        a: Linke Grenze
        b: Rechte Grenze
        n: Teilintervalle
        *args: Argumente der Funktion f

    Returns:
        float: Näherung für das Integral
    """
    h = (b - a) / n
    x_k = linspace(a, b, n+1)
    f_x_k = f(x_k[:-1], *args)
    return sum(f_x_k) * h


def int_trapez(f, a, b, n, *args):
    """
    Numerische Bestimmung des Integrals mit Hilfe der Simpson-Regel.

    Parameters:
        f: Zu integrierende Funktion f(x,*args)
        a: Linke Grenze
        b: Rechte Grenze
        n: Teilintervalle
        *args: Additional arguments to pass to the function f.

    Returns:
        float: The approximate integral of f from xl to xr.
    """
    h = (b-a)/n
    x_k = linspace(a, b, n+1)
    f_x_k = f(x_k, *args)
    f_x_k[0] *= 0.5 # Halbiere Randwerte, um sie mit in die Summe zu schreiben
    f_x_k[-1] *= 0.5
    return 0.5*h*(2*sum(f_x_k))


def int_simp(f, a, b, n, *args):
    """
    Integriere eine Funktion mit Hilfe der Simpson-Regel

    Parameter:
        f: Zu integrierende Funktion f(x,*args)
        a: Untergrenze
        b: Obergrenze
        n: Teilintervalle
        *args: Funktionsargumente für f

    Rückgabewert:
        float: Das Integral von f zwischen a und b
    """
    if n % 2 == 1:
        n += 1
    x_k = linspace(a, b, n + 1)
    f_x_k = f(x_k, *args)
    h = (b - a) / n
    return (h / 3) * (f_x_k[0] + 4 * sum(f_x_k[1:-1:2]) + 2 * sum(f_x_k[2:-2:2]) + f_x_k[-1] )


def nullstelle_bis(f, xl, xr, eps, *args):
    """
    Bestimme die Nullstelle einer Funktion mit Hilfe des Bisektionsverfahrens
    
    Parameter:
        f: Eine Funktion f(x,*args)
        xl: Linke Grenze 
        xr: Rechte Grenze
        eps: Zielgenauigkeit
        *args: Zusätzliche Argumente für die Funktion f.
    
    Rückgabewerte:
        x0: Nullstelle der Funktion
        steps: Schrittzahl
    """
    steps_max = 100
    fl = f(xl, *args)
    fr = f(xr, *args)
    
    if fl * fr > 0:
        raise ValueError("Function does not have opposite signs at the interval endpoints.")
    
    for steps in range(1,steps_max):
        xm = 0.5*(xl + xr)
        
        if 0.5*(xr - xl) < eps:
            return xm, steps
        
        fm = f(xm, *args) 

        if fm == 0.:
            return xm  # xm is a root of the function
        elif fl * fm < 0.:
            xr = xm
            fr = fm
        else:
            xl = xm
            fl = fm
    
    return 0.5*(xl + xr), steps_max


def nullstelle_new(f, df, x0, eps, *args):
    """
    Bestimme die Nullstelle einer Funktion mit Hilfe des Newton-Verfahrens
    
    Parameter:
        f: Eine Funktion f(x,*args)
        df: Ableitung der Funktion f
        x0: Startwert 
        eps: Zielgenauigkeit
        *args: Zusätzliche Argumente für die Funktionen f und df.
    
    Rückgabewerte:
        x0: Nullstelle der Funktion
        steps: Schrittzahl
    """
    steps_max = 100
    for steps in range(1,steps_max):
        fx0 = f(x0, *args)
        dfx0 = df(x0, *args)       
#        if dfx0 == 0:
#            raise ValueError("Ableitung ist 0. ")
        
        x1 = x0 - fx0 / dfx0

        if abs(x1 - x0) < eps:
            return x1,steps

        x0 = x1


def awp_euler(f, t_span, y0, h, *args):
    """
    Löse ein Anfangswertproblem y'(t) = f(t,y) mit y(t_0)=y_0 
    mit Hilfe der Euler-Cauchy-Methode

    Parameters:
        f: DGL-Term als Python-Funktion f(t,y,*args)
        t_span: Liste oder Tupel mit (t_0,t_ende)
        y0: y(t_0)
        h: Schrittweite
        *args: Argumente der Funktion f

    Returns:
        t: Stützstellen, an denen y rekonstruiert wurde
        y: Rekonstruierte Funktionswerte y
    """
    t = arange(t_span[0], t_span[1] + h, float(h))
    y = zeros_like(t)
    n_max = len(t)
    y[0] = y0
    
    for n in range(0, n_max-1):
        k1 = f(t[n], y[n], *args)
        y[n+1] = y[n] + h * k1
    
    return t, y


def awp_mid(f, t_span, y0, h, *args):
    """
    Löse ein Anfangswertproblem y'(t) = f(t,y) mit y(t_0)=y_0 
    mit Hilfe der Midpoint-Methode

    Parameters:
        f: DGL-Term als Python-Funktion f(t,y,*args)
        t_span: Liste oder Tupel mit (t_0,t_ende)
        y0: y(t_0)
        h: Schrittweite
        *args: Argumente der Funktion f

    Returns:
        t: Stützstellen, an denen y rekonstruiert wurde
        y: Rekonstruierte Funktionswerte y
    """
    t = arange(t_span[0], t_span[1] + h, float(h))
    y = zeros_like(t)
    n_max = len(t)
    y[0] = y0
    
    for n in range(0, n_max-1):
        k1 = f(t[n], y[n], *args)
        k2 = f(t[n] + 0.5*h, y[n] + 0.5*h*k1, *args)
        y[n+1] = y[n] +h*k2
    
    return t, y


def awp_rk4(f, t_span, y0, h, *args):
    """
    Löse ein Anfangswertproblem y'(t) = f(t,y) mit y(t_0)=y_0 
    mit Hilfe des Runge-Kutta-Verfahrens 4. Ordnung

    Parameters:
        f: DGL-Term als Python-Funktion f(t,y,*args)
        t_span: Liste oder Tupel mit (t_0,t_ende)
        y0: y(t_0)
        h: Schrittweite
        *args: Argumente der Funktion f

    Returns:
        t: Stützstellen, an denen y rekonstruiert wurde
        y: Rekonstruierte Funktionswerte y
    """
    t = arange(t_span[0], t_span[1] + h, float(h))
    y = zeros_like(t)
    n_max = len(t)
    y[0] = y0
    
    for n in range(0, n_max-1):
        k1 = f(t[n], y[n], *args)
        k2 = f(t[n] + 0.5 * h, y[n] + 0.5 * h * k1, *args)
        k3 = f(t[n] + 0.5 * h, y[n] + 0.5 * h * k2, *args)
        k4 = f(t[n] + h, y[n] + h * k3, *args)
        y[n+1] = y[n] + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    
    return t, y


def awp_vec_rk4(fvec, t_span, y0vec, h, *args):
    """
    Löse ein System von DGLs 1. Ordnung  
    mit Hilfe des Runge-Kutta-Verfahrens 4. Ordnung

    Parameters:
        f: Liste von DGL-Termen als Python-Funktion f_i(t,y,*args)
        t_span: Liste oder Tupel mit (t_0,t_ende)
        y0: Numpy-Array mit den Anfangsbedingungeny y_i(t_0)
        h: Schrittweite
        *args: Argumente der Funktionen f_i

    Returns:
        t: Stützstellen, an denen y rekonstruiert wurde
        y: Matrix der rekonstruierten Funktionswerte y
    """
    N = len(fvec)
    t = arange(t_span[0], t_span[1] + h, float(h))
    n_max = len(t)
    y = zeros((N,n_max))

    y[:,0] = y0vec
    
    for n in range(0, n_max-1):
        tev = t[n]
        yev = y[:,n]
        k1 = array([f(tev, yev, *args) for f in fvec])
        tev = t[n] + 0.5 * h
        yev = y[:,n] + 0.5 * h * k1
        k2 = array([f(tev, yev, *args) for f in fvec])
        tev = t[n] + 0.5 * h
        yev = y[:,n] + 0.5 * h * k2
        k3 = array([f(tev, yev, *args) for f in fvec])
        tev = t[n] + h
        yev = y[:,n] + h * k3
        k4 = array([f(tev, yev, *args) for f in fvec])
        y[:,n+1] = y[:,n] + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    
    return t, y