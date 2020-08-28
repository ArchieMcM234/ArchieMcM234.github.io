#include <SFML/Graphics.hpp>
#include <cmath>
#include <iostream>


class celestialBody {
    sf::Vector2f position;
    sf::Vector2f velocitee;
    sf::Vector2f acceleration;
    int mass;
    int radius; // both these could be set to floats
    
public:
    sf::CircleShape sfCircle;
    void updatePosition();
    void calcAccel(celestialBody planets[], int current, int numberPlanets);
    void setValues(int radiusTemp, int massTemp, sf::Vector2f positionTemp);
    void moveMouse(sf::Vector2i mousePos);
    
};

void celestialBody::moveMouse(sf::Vector2i mousePos) {
    position.x = (float)mousePos.x;
    position.y = (float)mousePos.y;
    velocitee = sf::Vector2f (0, 0);
    sfCircle.setPosition(position);
}

void celestialBody::setValues(int radiusTemp, int massTemp, sf::Vector2f positionTemp) {
    position = positionTemp;
    mass = massTemp;
    radius = radiusTemp;
    sfCircle.setFillColor(sf::Color::Green);
    sfCircle.setOrigin(sf::Vector2f(radius, radius));
    sfCircle.setRadius(radius);
    sfCircle.setPosition(position);


}

void celestialBody::updatePosition() {
    velocitee += acceleration;
    position += velocitee;
    sfCircle.setPosition(position);
}

void celestialBody::calcAccel(celestialBody planets[], int current, int numberPlanets) {
    acceleration = sf::Vector2f(0, 0);
    for (int i = 0; i < numberPlanets; i++) {
        if (!(i == current)) {
            float hypotenuse = sqrt(pow(planets[i].position.x - position.x, 2) + pow(planets[i].position.y - position.y, 2));
            acceleration.x += ((planets[i].position.x - position.x)*planets[i].mass * 0.0001)/pow(hypotenuse, 3);
            acceleration.y += ((planets[i].position.y - position.y) * planets[i].mass * 0.0001) / pow(hypotenuse, 3);
        }
    }

}

void drawPlanets(sf::RenderWindow& window, celestialBody planets[], int numberPlanets) {
    for (int i = 0; i < numberPlanets; i++) {
        window.draw(planets[i].sfCircle);
    }
}


int main()
{
    const int numberPlanets = 2;
    sf::RenderWindow window(sf::VideoMode(1700, 900), "Gravity Simulation", sf::Style::Default);
    celestialBody planets [numberPlanets];

    // define planets if wanted can implemented with text input
    planets[0].setValues(20, 1000, sf::Vector2f(750, 300));
    planets[1].setValues(20, 10, sf::Vector2f(250, 200));
    //planets[2].setValues(20, 1000, sf::Vector2f(500, 400));

    // game loop
    while (window.isOpen())
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();
        }




        // ################# update postions
        for (int i = 0; i < numberPlanets; i++) {
            planets[i].calcAccel(planets, i, numberPlanets);
        }
        for (int i = 0; i < numberPlanets; i++) {
            planets[i].updatePosition();
        }
        // #################
        if (sf::Mouse::isButtonPressed(sf::Mouse::Left)) {
            planets[0].moveMouse(sf::Mouse::getPosition(window));
        }
        // ################# draw results to screen
       
        window.clear(); 
        drawPlanets(window, planets, numberPlanets);
        window.display();

        // ################# 
    }

    return 0;
}