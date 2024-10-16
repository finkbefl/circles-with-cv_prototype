# Class to plot data with bokeh

# Import the external packages
# Operating system functionalities
import sys
import os
# Plotting with bokeh
from bokeh.plotting import figure, show
from bokeh.io import output_file
from bokeh.layouts import gridplot, column
from bokeh.palettes import Category10_10
from pandas.core.series import Series
#from bokeh.plotting.figure import Figure    # requires bokeh version 2.4.3, now using version 3
from bokeh.models import Range1d, BoxAnnotation, Label
from numbers import Real
# Import internal packages/ classes
from utils.own_logging import OwnLogging
from utils.check_parameter import checkParameter, checkParameterString
from utils.own_exceptions import IllegalArgumentError

# Create own color palette: based on colors from latex (find out with gimp color-picker) to have a common color base for the documentation
ownColorPalette = [
    '#008080', # category10: teal
    '#ff8000', # category10: orange
    '#800080', # category10: violet
    '#2ca02c', # category10: green
    '#d62728', # category10: red
    '#8c564b', # category10: brown
    '#1f77b4', # category10: blue
    '#7f7f7f', # category10: gray
    '#bcbd22', # category10: lime
    '#17becf'  # category10: cyan
    ] 

class PlotBokeh():
    """
    A basic class for easy plotting with bokeh and output the result in a static HTML file if required
    ----------
    Attributes:
        file_name : str
            The file name (html) in which the figure is shown
        file_title : str
            The output file title
    ----------
    Methods
        no methods
    """

    # Initialize the logger
    __own_logging = OwnLogging(__name__)
    __own_logger = __own_logging.logger

    # Constructor Method
    def __init__(self, file_name=None, file_title=None):
        # Output to static HTML file
        if file_name is not None and file_title is not None:
            checkParameterString(file_name)
            checkParameterString(file_title)
            # Create directory if it not exist
            if not os.path.exists(os.path.dirname(file_name)):
                os.makedirs(os.path.dirname(file_name))
            output_file(file_name, title=file_title)
            self.__own_logger.info("Bokeh plot initialized for output file %s", file_name)
        elif file_name is None and file_title is None:
            # No output file, used when a inherited class is used to fill a complex visualization file --> A valid state!
            pass
        else:
            raise IllegalArgumentError("A parameter cannot be None")

class PlotMultipleLayers(PlotBokeh):
    """
    A class for easy plotting multiple layers with bokeh and output the result in a static HTML file if required
    ----------
    Attributes:
        figure_title : str
            The title of the figure
        x_label : str
            The label of the x axis
        y_label : str
            The label of the y axis
        x_axis_type : str
            The type of the x-axis
        x_range : range
            The range of the x-axis for categorical data
        file_name : str
            The file name (html) in which the figure is shown
        file_title : str
            The output file title
    ----------
    Methods
        addCircleLayer(legend_label, x_data, y_data):
            Add a circle layer to the figure with the given data
        addLineCircleLayer(legend_label, x_data, y_data):
            Add a circle and line layer to the figure with the given data
        addVBarLayer(legend_label, x_data, y_data):
            Add a vertical bar layer to the figure with the given data
        addHist(edges, hist):
            Add a histogram layer to the figure
        add_green_box(top_val):
            Add a green box from y=0 up to the specified y value
        add_vertical_line(x_pos, top_val, bottom_val=0):
            Add a vertical line
        add_annotation(x_pos, y_pos, text):
            Add a text
        getFigure():
            Get the figure
        showPlot():
            show the figure
    """

    # Initialize the logger
    __own_logging = OwnLogging(__name__)
    __own_logger = __own_logging.logger

    # Constructor Method
    def __init__(self, figure_title, x_label, y_label, x_axis_type='auto', x_range=None, file_name=None, file_title=None):
        # Call the Base Class Constructor
        PlotBokeh.__init__(self, file_name, file_title)
        # For color cycling (different colors for the different layers)
        self.__color_iter = ownColorPalette.__iter__()
        # create a figure, but first check the parameter
        checkParameterString(figure_title)
        #checkParameterString(x_label)
        #checkParameterString(y_label)
        if x_range is None:
            # Figure without explicit x_range defined (e.g. for histogram or for datetime as x_axis)
            self.__own_figure = figure(title=figure_title, x_axis_type=x_axis_type, x_axis_label=x_label, y_axis_label=y_label)
        else:
            # Figure with explicit x_range (for the case that the bounds of the x-data cannot be determined automatically e.g. for categorical data)
            self.__own_figure = figure(title=figure_title, x_axis_type=x_axis_type, x_range=x_range, x_axis_label=x_label, y_axis_label=y_label)
        self.__own_logger.info("Bokeh plot for multiple layers initialized for figure %s", figure_title)

    def addLineCircleLayer(self, legend_label, x_data, y_data, legend_location='top_right'):
        """
        Add a layer to the figure (line and circle representation)
        ----------
        Parameters:
            legend_label : str
                The legend label of the layer
            x_data : Series
                The x data to plot
            y_data : Series
                The y data to plot
            legend_location : str
                The location of the legend
        ----------
        Returns:
            no returns
        """
        # check the parameter
        checkParameterString(legend_label)
        #checkParameter(x_data, Series)
        #checkParameter(y_data, Series)
        # Assign the next color automatically from the color iterator
        color = next(self.__color_iter)
        # add the plots to the figure
        self.__own_figure.line(x=x_data, y=y_data, legend_label=legend_label, color=color)
        self.__own_figure.circle(x=x_data, y=y_data, legend_label=legend_label, color=color)
        self.__own_figure.legend.location=legend_location
        self.__own_logger.info("Added  line/circle layer %s", legend_label)

    def addCircleLayer(self, legend_label, x_data, y_data):
        """
        Add a layer to the figure (circle representation)
        ----------
        Parameters:
            legend_label : str
                The legend label of the layer
            x_data : Series
                The x data to plot
            y_data : Series
                The y data to plot
        ----------
        Returns:
            no returns
        """
        # check the parameter
        checkParameterString(legend_label)
        checkParameter(x_data, Series)
        checkParameter(y_data, Series)
        # add a plot to the figure, assign the next color automatically from the color iterator
        self.__own_figure.circle(x=x_data, y=y_data, legend_label=legend_label, color=next(self.__color_iter))
        self.__own_logger.info("Added  circle layer %s", legend_label)

    def addVBarLayer(self, x_data, y_data, color_sequencing=True, legend_label=None, width=0.8, fill_alpha=0.6, legend_location='top_right'):
        """
        Add a layer to the figure (vertical bar representation)
        ----------
        Parameters:
            x_data : numbers.Real
                The x data to plot
            y_data : numbers.Real
                The y data to plot
            color_sequencing : boolean
                A flag, whether every bar sould be drawn in another color with the known color sequence
            legend_label : str
                The legend label of the layer
            width : numbers.Real
                The width of the bar
            fill_alpha : numbers.Real
                Opacity of the filling colour
            legend_location : str
                The location of the legend
        ----------
        Returns:
            no returns
        """
        # check the parameter
        #checkParameter(x_data, Real)
        #checkParameter(y_data, Real)
        if color_sequencing:
            # add a plot to the figure, assign every bar in another color with the known color sequence
            if legend_label is None:
                self.__own_figure.vbar(x = x_data, top = y_data, width=width, color=ownColorPalette[0:len(x_data)])
            # else :
            #     # When legend_labels are defined, then we assume that we need multiple layers  with different colors and the bars should be visible with transparent colors to get overlaps visible
            #     # As this option allows color-sequencing, multiple layers makes no sense?
        else:
            # add a plot to the figure
            if legend_label is None:
                self.__own_figure.vbar(x = x_data, top = y_data, width=width, color=next(self.__color_iter))
                self.__own_figure.legend.location=legend_location
            else :
                # When legend_labels are defined, then we assume that we need multiple layers  with different colors and the bars should be visible with transparent colors to get overlaps visible
                self.__own_figure.vbar(x = x_data, top = y_data, width=width, legend_label=legend_label, color=next(self.__color_iter), fill_alpha=fill_alpha)
                self.__own_figure.legend.location=legend_location

        self.__own_logger.info("Added vbar layer")

    def addHist(self, edges, hist, legend_label=None):
        """
        Add a histogram layer to the figure
        ----------
        Parameters:
        edges : numbers.Real
            The bins edges data to plot
        hist : numbers.Real
            The histogram data to plot
        legend_label : str
                The legend label of the layer
        ----------
        Returns:
            no returns
        """
        # check the parameter
        #checkParameter(x_data, Real)
        #checkParameter(y_data, Real)
        # add a plot to the figure
        if legend_label is None:
            self.__own_figure.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], line_color="white", color=next(self.__color_iter))
        else :
            # When legend_labels are defined, then we assume that we need multiple layers  with different colors and the bars should be visible with transparent colors to get overlaps visible
            self.__own_figure.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], line_color="white", legend_label=legend_label, color=next(self.__color_iter), fill_alpha=0.6)
        self.__own_logger.info("Added hist layer")

    def set_axis_range(self, x_min=None, x_max=None, y_min=None, y_max=None):
        """
        Set the axis range of the figure
        ----------
        Parameters:
            x_min : numbers.Real
                The min x value
            x_max : numbers.Real
                The max x value
            y_min : numbers.Real
                The min y value
            y_max : numbers.Real
                The max y value
        ----------
        Returns:
            no returns
        """
        checkParameter(x_min, Real, True)
        checkParameter(x_max, Real, True)
        checkParameter(y_min, Real, True)
        checkParameter(y_max, Real, True)
        if x_min is not None and x_max is not None and y_min is not None and y_max is not None:
            self.__own_figure.x_range = Range1d(x_min, x_max)
            self.__own_figure.y_range = Range1d(y_min, y_max)
        elif x_min is not None and x_max is not None and y_min is None and y_max is None:
            self.__own_figure.x_range = Range1d(x_min, x_max)
        elif y_min is not None and y_max is not None and x_min is None and x_max is None:
            self.__own_figure.y_range = Range1d(y_min, y_max)
        else:
            raise IllegalArgumentError("A range must be a value pair!")

    def add_green_box(self, top_val, bottom_val=0):
        """
        Add a green box from y=0 up to the specified y value
        ----------
        Parameters:
            top_val : numbers.Real
                The upper y value as limit for the color box
            bottom_val : numbers.Real
                The lower y value as limit for the color box
        ----------
        Returns:
            no returns
        """
        checkParameter(top_val, Real, False)
        green_box = BoxAnnotation(top=top_val, bottom=bottom_val, fill_alpha=0.1, fill_color="green")
        self.__own_figure.add_layout(green_box)
    
    def add_vertical_line(self, x_pos, top_val, bottom_val=0):
        """
        Add a vertical line at the x_pos from y=0 up to the specified y value
        ----------
        Parameters:
            x_pos : numbers.Real
                The x position of the line
            top_val : numbers.Real
                The upper y position of the line
            bottom_val : numbers.Real
                The lower y position of the line
        ----------
        Returns:
            no returns
        """
        checkParameter(x_pos, Real, False)
        checkParameter(top_val, Real, False)
        self.__own_figure.line([x_pos, x_pos], [bottom_val, top_val], line_width=2, color="black")

    def add_annotation(self, x_pos, y_pos, text, text_align="center"):
        """
        Add a text at the specific postition
        ----------
        Parameters:
            x_pos : numbers.Real
                The x position
            y_pos : numbers.Real
                The y position
            text : str
                The text
            text_align : str
                The position alignment of the text
        ----------
        Returns:
            no returns
        """
        checkParameterString(text)
        checkParameter(x_pos, Real, False)
        checkParameter(y_pos, Real, False)
        mytext = Label(x=x_pos, y=y_pos, text=text, text_align=text_align, )
        self.__own_figure.add_layout(mytext)

    def getFigure(self):
        """
        Get the figure
        ----------
        Parameters:
            no parameter
        ----------
        Returns:
            Returns the figure (bokeh.plotting.figure.Figure)
        """
        return self.__own_figure

    def showPlot(self):
        """
        Show the plot
        ----------
        Parameters:
            no parameter
        ----------
        Returns:
            Show the plot
        """
        # show the figure
        self.__own_logger.info("Show the plot")
        show(self.__own_figure)

    def showPlotResponsive(self, sizing_mode='stretch_both'):
        """
        Show the plot in column layout (responsive)
        ----------
        Parameters:
            sizing_mode : Str
                The sizing mode (default: stretch both -> completely responsive)
        ----------
        Returns:
            Show the plot
        """
        # Set the sizing_mode for all figures in list
        for figure in self.__figure_list:
            figure.sizing_mode=sizing_mode
        self.__own_logger.info("Show the column layout")
        plot = column(self.__figure_list, sizing_mode=sizing_mode)

class PlotMultipleFigures(PlotBokeh):
    """
    A class for easy plotting multiple figures with bokeh and output the result in a static HTML file
    ----------
    Attributes:
        file_name : str
            The file name (html) in which the figure is shown
        file_title : str
            The output file title
    ----------
    Methods
        appendFigure(figure):
            Add a figure (bokeh.plotting.figure.Figure) to the plot
        showPlot():
            show the plot
    """

    # Initialize the logger
    __own_logging = OwnLogging(__name__)
    __own_logger = __own_logging.logger

    # Constructor Method
    def __init__(self, file_name, file_title):
        # Call the Base Class Constructor
        PlotBokeh.__init__(self, file_name, file_title)
        # The list of figures, which will be plotted
        self.__figure_list = []
        self.__own_logger.info("Bokeh plot for multiple figures initialized for file name %s", file_name)

    def appendFigure(self, figure):
        """
        Add a fiigure to the plot
        ----------
        Parameters:
            figure : bokeh.plotting.figure.Figure
                The figure to add
        ----------
        Returns:
            no returns
        """
        # check the parameter
        #checkParameter(figure, Figure)
        self.__figure_list.append(figure)
        self.__own_logger.info("Appendend figure to Bokeh plot %s", figure)

    def showPlot(self, ncols=2, plot_width=None, plot_height=None):
        """
        Show the plot in gridplot layout (not responsive, fixed sizes in pixel)
        ----------
        Parameters:
            ncols : int
                The number of columns (default: 2)
            plot_width : int
                The plot width (default: None)
            plot_height : int
                The plot height (default: None)
        ----------
        Returns:
            Show the plot
        """
        self.__own_logger.info("Show the gridplot layout")
        plot = gridplot(self.__figure_list, ncols=ncols, plot_width=plot_width, plot_height=plot_height)
        show(plot)

    def showPlotResponsive(self, sizing_mode='stretch_both'):
        """
        Show the plot in column layout (responsive)
        ----------
        Parameters:
            sizing_mode : Str
                The sizing mode (default: stretch both -> completely responsive)
        ----------
        Returns:
            Show the plot
        """
        # Set the sizing_mode for all figures in list
        for figure in self.__figure_list:
            figure.sizing_mode=sizing_mode
        self.__own_logger.info("Show the column layout")
        #plot = column(self.__figure_list, sizing_mode=sizing_mode)
        plot = column(self.__figure_list, sizing_mode='fixed', width=700, height=400) # For documentation purposes
        show(plot)

#########################################################

def figure_vbar(logger, figure_title, y_label, x_data, y_data, set_x_range=True, color_sequencing=True, x_label=None, width=0.8):
    """
    Function to create a vbar chart figure
    ----------
    Parameters:
        logger : Logger
            The Logger to log with
        figure_title : str
            The title of the figure
        x_label : str
            The label of the x axis
        y_label : str
            The label of the y axis
        x_data : numbers.Real
            The x data to plot
        y_data : numbers.Real
            The y data to plot
        set_x_range : boolean
            set the x_data as range of the x-axis (for categorical data)
        color_sequencing : boolean
            A flag, whether every bar sould be drawn in another color with the known color sequence
        width : numbers.Real
                The width of the bar
    ----------
    Returns:
        The bokeh class
    """

    try:
        logger.info("Figure for vbar chart: %s", figure_title)
        # Set the x_data as x_range (for categorical data)?
        if set_x_range:
            figure = PlotMultipleLayers(figure_title, x_label, y_label, x_range=x_data)
        # Dont set the x_range
        else:
            figure = PlotMultipleLayers(figure_title, x_label, y_label, x_range=None)
        figure.addVBarLayer(x_data, y_data, color_sequencing=color_sequencing, width=width)
        return figure
    except TypeError as error:
        logger.error("########## Error when trying to create figure ##########", exc_info=error)
        sys.exit('A parameter does not match the given type')

#########################################################

def figure_vbar_as_layers(logger, figure_title, y_label, layers, x_data, y_data, set_x_range=True, x_label=None, width=0.8, single_x_range=False, fill_alpha=0.6, legend_location='top_right', x_offset=0):
    """
    Function to create a vbar chart figure
    ----------
    Parameters:
        logger : Logger
            The Logger to log with
        figure_title : str
            The title of the figure
        x_label : str
            The label of the x axis
        y_label : str
            The label of the y axis
        layers : array
            The names of the layers
        x_data : numbers.Real
            The x data to plot
        y_data : numbers.Real
            The y data to plot
        set_x_range : boolean
            set the x_data as range of the x-axis (for categorical data)
        width : numbers.Real
                The width of the bar
        single_x_range : boolean
                Use single x_range for multiple layers
        fill_alpha : numbers.Real
                Opacity of the filling colour
        legend_location : str
            The location of the legend
        x_offset : numbers.Real
            The x-offset for visualisation of layers next to each other and no overlay
        
    ----------
    Returns:
        The bokeh class
    """

    try:
        logger.info("Figure for vbar chart: %s", figure_title)
        # Set the x_data as x_range (for categorical data)?
        if set_x_range:
            figure = PlotMultipleLayers(figure_title, x_label, y_label, x_range=x_data)
        # Dont set the x_range
        else:
            figure = PlotMultipleLayers(figure_title, x_label, y_label, x_range=None)
        for (index, layer) in enumerate(layers):
            logger.info("Add Layer for %s", layer)
            if single_x_range:
                offsets = map(lambda x: x_offset*index-x_offset, x_data)
                figure.addVBarLayer(list(zip(x_data, offsets)), y_data[index], color_sequencing=False, legend_label=layer, width=width, fill_alpha=fill_alpha, legend_location=legend_location)
            else:
                figure.addVBarLayer(x_data[index], y_data[index], color_sequencing=False, legend_label=layer, width=width, fill_alpha=fill_alpha, legend_location=legend_location)
        return figure
    except TypeError as error:
        logger.error("########## Error when trying to create figure ##########", exc_info=error)
        sys.exit('A parameter does not match the given type')

#########################################################

def figure_hist(logger, figure_title, x_label, y_label, edges, hist):
    """
    Function to create a histogram chart figure
    ----------
    Parameters:
        logger : Logger
            The Logger to log with
        figure_title : str
            The title of the figure
        x_label : str
            The label of the x axis
        y_label : str
            The label of the y axis
        edges : numbers.Real
            The bins edges data to plot
        hist : numbers.Real
            The histogram data to plot
    ----------
    Returns:
        The bokeh class
    """

    try:
        logger.info("Figure for hist chart: %s", figure_title)
        figure = PlotMultipleLayers(figure_title, x_label=x_label, y_label=y_label)
        figure.addHist(edges, hist)
        return figure
    except TypeError as error:
        logger.error("########## Error when trying to create figure ##########", exc_info=error)
        sys.exit('A parameter does not match the given type')

#########################################################
        
def figure_hist_as_layers(logger, figure_title, x_label, y_label, layers, edges, hists):
    """
    Function to create a histogram chart figure as multiple layers
    ----------
    Parameters:
        logger : Logger
            The Logger to log with
        figure_title : str
            The title of the figure
        x_label : str
            The label of the x axis
        y_label : str
            The label of the y axis
        layers : array
            The names of the layers
        edges : DataFrame
            The bins edges data to plot
        hists : DataFrame
            The histogram data to plot
    ----------
    Returns:
        The bokeh class
    """

    try:
        logger.info("Figure for hist chart: %s", figure_title)
        figure = PlotMultipleLayers(figure_title, x_label=x_label, y_label=y_label)
        for (index, layer) in enumerate(layers):
            logger.info("Add Layer for %s", layer)
            figure.addHist(edges[index], hists[index], layer)
        return figure
    except TypeError as error:
        logger.error("########## Error when trying to create figure ##########", exc_info=error)
        sys.exit('A parameter does not match the given type')

#########################################################

def figure_time_series_data_as_layers(logger, figure_title, y_label, x_data, y_layers, y_datas, x_label=None, set_x_range=False, x_axis_type='auto', legend_location='top_right'):
    """
    Function to create a figure for time series data as multiple layers
    ----------
    Parameters:
        logger : Logger
            The Logger to log with
        figure_title : str
            The title of the figure
        y_label : str
            The label of the y axis
        x_data : Series
                The x data to plot
        y_layers : array
            The names of the layers
        y_datas : list of Series
            The y data to plot
        x_label : str
            The label of the x axis
        set_x_range : Boolean
            Set the x_data as x_range when creating a figure (for categorical data)
        x_axis_type : str
            The type of the x-axis
        legend_location : str
            The location of the legend
    ----------
    Returns:
        The figure
    """

    try:
        logger.info("Figure for times series data as multiple layers with title %s", figure_title)
        if set_x_range:
            figure = PlotMultipleLayers(figure_title, x_label, y_label, x_range=x_data)
        else:
            figure = PlotMultipleLayers(figure_title, x_label, y_label, x_axis_type=x_axis_type)
        for (index, layer) in enumerate(y_layers):
            logger.info("Add Layer for %s", layer)
            figure.addLineCircleLayer(layer, x_data, y_datas[index], legend_location)
        return figure
    except TypeError as error:
        logger.error("########## Error when trying to create figure ##########", exc_info=error)
        sys.exit('A parameter does not match the given type')
