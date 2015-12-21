var figureWidth = 400;  // Height of the entire figure including labels and title.
var figureHeight = 400;  // Width of the entire figure including labels and title.
var figureMargin = {top: 10, right: 10, bottom: 10, left: 10};  // Margin around each individual figure.
var svgWidth = 2 * (figureWidth + figureMargin.left + figureMargin.right);  // Width of the SVG element needed to hold both figures and their padding.
var svgHeight = 2 * (figureWidth + figureMargin.top + figureMargin.bottom);  // Height of the SVG element needed to hold both figures and their padding.
var svgMargin = {top: 10, right: 10, bottom: 10, left: 10};  // The margin around the set of figures.

// Create the SVG element.
var svg = d3.select("body")
    .append("svg")
        .attr("width", svgWidth + svgMargin.left + svgMargin.right)
        .attr("height", svgHeight + svgMargin.top + svgMargin.bottom)
    .append("g")
        .attr("transform", "translate(" + svgMargin.left + ", " + svgMargin.top + ")");

// Read the data in.
var dataAccessorFunction = function(d)
    {
        return {First : +d.FirstModel, Second : +d.SecondModel};
    }
d3.tsv("/Data/DisambiguationResults.tsv", dataAccessorFunction, function(error, data)
    {
        // Sort the data by the value of the first model.
        data = data.sort(function(a, b) { return d3.ascending(a.First, b.First); });

        // Create the figure the type 1 vs type 2 diabetes results.
        var figure1v2 = svg.append("g")
            .attr("transform", "translate(0, 0)");
        createFigure(figure1v2, data.map(function(d) { return d.First; }));

        // Create the figure for the diabetes vs non-diabetes results.
        var figureDvsND = svg.append("g")
            .attr("transform", "translate(" + (svgWidth / 2) + ", 0)");
        createFigure(figureDvsND, data.map(function(d) { return d.Second; }));
    }
);

function createFigure(figureContainer, dataArray)
{
}