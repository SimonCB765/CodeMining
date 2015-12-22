var figureWidth = 400;  // Height of the entire figure including labels and title.
var figureHeight = 400;  // Width of the entire figure including labels and title.
var figureMargin = {top: 10, right: 10, bottom: 10, left: 10};  // Margin around each individual figure.
var svgWidth = 2 * (figureWidth + figureMargin.left + figureMargin.right);  // Width of the SVG element needed to hold both figures and their padding.
var svgHeight = figureHeight + figureMargin.top + figureMargin.bottom;  // Height of the SVG element needed to hold both figures and their padding.
var svgMargin = {top: 10, right: 10, bottom: 50, left: 60};  // The margin around the set of figures.

// Create the SVG element.
var svg = d3.select("body")
    .append("svg")
        .attr("width", svgWidth + svgMargin.left + svgMargin.right)
        .attr("height", svgHeight + svgMargin.top + svgMargin.bottom)
    .append("g")
        .attr("transform", "translate(" + svgMargin.left + ", " + svgMargin.top + ")");

//REMOVE
svg.append("rect").attr("x", -svgMargin.left).attr("y", -svgMargin.top).attr("height", svgHeight + svgMargin.top + svgMargin.bottom).attr("width", svgWidth + svgMargin.left + svgMargin.right).style("opacity", 0.1);
svg.append("rect").attr("height", svgHeight).attr("width", svgWidth).style("opacity", 0.1);
//REMOVE

// Read the data in.
var dataAccessorFunction = function(d)
    {
        // If there is no entry on a line (because the variable was not used in the model), then the returned value for the coefficient will be 0.
        return {DisambiguationFirstModel : Math.abs(+d.DisambiguationFirstModel), DisambiguationSecondModel : Math.abs(+d.DisambiguationSecondModel),
                DiabetesFirstModel : Math.abs(+d.DiabetesFirstModel), DiabetesSecondModel : Math.abs(+d.DiabetesSecondModel)};
    }
d3.tsv("/Data/CoefComparisonResults.tsv", dataAccessorFunction, function(error, data)
    {
    console.log(data);
        // Create the figure the type 1 vs type 2 diabetes results.
        var data1v2 = data.sort(function(a, b) { return d3.descending(a.DisambiguationFirstModel, b.DisambiguationFirstModel); });  // Sort the data by the value of the first model.
        var figure1v2 = svg.append("g")
            .attr("transform", "translate(" + figureMargin.left + ", " + figureMargin.top + ")");
//REMOVE
        figure1v2.append("rect").attr("height", figureHeight).attr("width", figureWidth).style("opacity", 0.1);
//REMOVE
        createFigure(figure1v2, data.map(function(d, index) { return {First : d.DisambiguationFirstModel, Second : d.DisambiguationSecondModel, index : index}; }), "Type 1 Vs. Type 2 Diabetes");

        // Create the figure for the diabetes vs non-diabetes results.
        var dataDvND = data.sort(function(a, b) { return d3.descending(a.DiabetesFirstModel, b.DiabetesFirstModel); });  // Sort the data by the value of the first model.
        var figureDvND = svg.append("g")
            .attr("transform", "translate(" + ((2 * figureMargin.left) + figureWidth + figureMargin.right) + ", " + figureMargin.top + ")");
//REMOVE
        figureDvND.append("rect").attr("height", figureHeight).attr("width", figureWidth).style("opacity", 0.1);
//REMOVE
        createFigure(figureDvND, data.map(function(d, index) { return {First : d.DiabetesFirstModel, Second : d.DiabetesSecondModel, index : index}; }), "Diabetes Vs. Non-diabetes");
    }
);

function createFigure(figureContainer, dataArray, figureTitle)
{
}